// AlphaZero MCTS - 使用神经网络指导的蒙特卡洛树搜索
#![cfg(feature = "alphazero")]

use super::alphazero_net::AlphaZeroNet;
use super::board::*;
use super::utils::*;
use std::collections::HashMap;
use tch::{Device, Kind, Tensor};

/// AlphaZero MCTS 节点
#[derive(Debug, Clone)]
struct AlphaNode {
    visit_count: u32,
    total_value: f32,
    prior_prob: f32,
    children: HashMap<(usize, usize), usize>, // (move) -> child_index
}

impl AlphaNode {
    fn new(prior_prob: f32) -> Self {
        Self {
            visit_count: 0,
            total_value: 0.0,
            prior_prob,
            children: HashMap::new(),
        }
    }

    fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f32
        }
    }

    fn ucb_score(&self, parent_visits: u32, c_puct: f32) -> f32 {
        let q = self.q_value();
        let u = c_puct * self.prior_prob * (parent_visits as f32).sqrt()
            / (1.0 + self.visit_count as f32);
        q + u
    }
}

/// AlphaZero MCTS
pub struct AlphaZeroMCTS {
    nodes: Vec<AlphaNode>,
    states: Vec<Board>,
    root_index: usize,
    c_puct: f32,
    num_simulations: u32,
}

impl AlphaZeroMCTS {
    pub fn new(root_state: Board, num_simulations: u32) -> Self {
        let root = AlphaNode::new(1.0);
        Self {
            nodes: vec![root],
            states: vec![root_state],
            root_index: 0,
            c_puct: 1.5,
            num_simulations,
        }
    }

    /// 列出当前状态下的所有合法走法（所有空位，不使用启发式裁剪）
    fn legal_moves(&self, state: &Board) -> Vec<(usize, usize)> {
        let gravity_mode = state.win_len == 4 && state.width == 7 && state.height == 6;
        let mut mvs = Vec::new();
        if gravity_mode {
            // 经典 Connect4：每列一个可落子的最低空位
            for j in 0..state.width {
                let mut placed = false;
                for i in (0..state.height).rev() {
                    if state.get(i as i32, j as i32) == Some(0) {
                        mvs.push((i, j));
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    // 此列已满，跳过
                }
            }
        } else {
            for i in 0..state.height {
                for j in 0..state.width {
                    if state.get(i as i32, j as i32) == Some(0) {
                        mvs.push((i, j));
                    }
                }
            }
        }
        mvs
    }

    /// 选择最佳子节点（UCB）
    fn select(&self, node_idx: usize) -> Option<(usize, usize, usize)> {
        let node = &self.nodes[node_idx];
        let parent_visits = node.visit_count;

        let mut best_move = None;
        let mut best_score = f32::MIN;
        let mut best_child_idx = 0;

        for (&mv, &child_idx) in &node.children {
            let child = &self.nodes[child_idx];
            let score = child.ucb_score(parent_visits, self.c_puct);

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
                best_child_idx = child_idx;
            }
        }

        best_move.map(|mv| (mv.0, mv.1, best_child_idx))
    }

    /// 扩展节点（使用神经网络预测策略和价值）
    fn expand(&mut self, node_idx: usize, net: &AlphaZeroNet, player: u8) -> f32 {
        // 当前状态
        let state = self.states[node_idx].clone();

        // 先检查是否为终局（胜/负/和）
        if let Some(winner) = state.any_winner() {
            return if winner == player { 1.0 } else { -1.0 };
        }
        if state.empty_cells_count() == 0 {
            return 0.0; // 和局
        }

        // 将棋盘转换为张量并进行网络预测
        let board_tensor = self.board_to_tensor(&state, player);
        let (policy_logp, value) = net.predict(&board_tensor);

        // 合法走法：所有空位
        let valid_moves = self.legal_moves(&state);
        if valid_moves.is_empty() {
            // 没有可走步 —— 视为和局（保守）
            return 0.0;
        }

        // 将 log_softmax 概率转为 softmax 概率，并仅取合法步，再做归一化
        let flat_size = BOARD_WIDTH * BOARD_HEIGHT;
        let policy_tensor = policy_logp
            .view([flat_size as i64])
            .to_kind(Kind::Float)
            .to_device(Device::Cpu);
        let mut priors: Vec<(usize, usize, f32)> = Vec::with_capacity(valid_moves.len());
        let mut sum_p = 0.0f32;
        for &(x, y) in &valid_moves {
            let idx = (x * BOARD_WIDTH + y) as i64;
            let lp = policy_tensor.get(idx).double_value(&[]) as f32;
            let p = lp.exp(); // 从 log_softmax 恢复到 softmax 概率
            priors.push((x, y, p));
            sum_p += p;
        }
        if sum_p <= 1e-8 {
            // 极端情况：网络将所有合法步概率压到 ~0，退化为均匀分布
            let unif = 1.0 / (priors.len() as f32);
            for e in &mut priors {
                e.2 = unif;
            }
        } else {
            for e in &mut priors {
                e.2 /= sum_p;
            }
        }

        // 如果是根节点，注入 Dirichlet 噪声增强探索（可选）
        #[cfg(feature = "random")]
        if node_idx == self.root_index {
            use rand::distributions::Distribution;
            use rand_distr::Gamma;
            let alpha = 0.3f32; // Connect4 尺度较小，噪声适中
            let eps = 0.25f32;
            let k = priors.len();
            if k > 0 {
                let gamma = Gamma::new(alpha as f64, 1.0).unwrap();
                let mut noise: Vec<f32> = (0..k)
                    .map(|_| gamma.sample(&mut rand::thread_rng()) as f32)
                    .collect();
                let sum_n: f32 = noise.iter().copied().sum();
                if sum_n > 0.0 {
                    for n in &mut noise {
                        *n /= sum_n;
                    }
                    for (i, e) in priors.iter_mut().enumerate() {
                        e.2 = (1.0 - eps) * e.2 + eps * noise[i];
                    }
                }
            }
        }

        // 创建子节点与子状态
        for (x, y, prior) in priors {
            let child = AlphaNode::new(prior);
            let child_idx = self.nodes.len();
            self.nodes.push(child);

            let mut child_state = state.clone();
            child_state.place(x, y, player);
            self.states.push(child_state);

            self.nodes[node_idx].children.insert((x, y), child_idx);
        }

        // 返回当前网络对该局面的评估值（从当前 player 的视角）
        value.double_value(&[]) as f32
    }

    /// 反向传播
    fn backpropagate(&mut self, path: &[usize], value: f32) {
        // 沿路径回传价值，层间轮换视角需要翻转符号
        let mut v = value;
        for &idx in path.iter().rev() {
            let node = &mut self.nodes[idx];
            node.visit_count += 1;
            node.total_value += v;
            v = -v;
        }
    }

    /// 执行一次模拟
    fn simulate(&mut self, net: &AlphaZeroNet, player: u8) -> f32 {
        let mut path = vec![self.root_index];
        let mut current_idx = self.root_index;
        let mut current_player = player;

        // 选择阶段：沿着树向下走
        loop {
            let node = &self.nodes[current_idx];

            // 终局检查：若已分出胜负或和局，直接回传
            {
                let state = &self.states[current_idx];
                if let Some(winner) = state.any_winner() {
                    let v = if winner == current_player { 1.0 } else { -1.0 };
                    self.backpropagate(&path, v);
                    return v;
                }
                if state.empty_cells_count() == 0 {
                    self.backpropagate(&path, 0.0);
                    return 0.0;
                }
            }

            // 如果是叶节点或未访问过，进行扩展
            if node.children.is_empty() {
                let value = self.expand(current_idx, net, current_player);
                self.backpropagate(&path, value);
                return value;
            }

            // 选择最佳子节点
            if let Some((_x, _y, child_idx)) = self.select(current_idx) {
                path.push(child_idx);
                current_idx = child_idx;
                current_player = cfg::opponent(current_player);
            } else {
                break;
            }
        }

        0.0 // 平局
    }

    /// 运行 MCTS 搜索
    pub fn search(&mut self, net: &AlphaZeroNet, player: u8) {
        for _ in 0..self.num_simulations {
            self.simulate(net, player);
        }
    }

    /// 获取搜索策略（访问计数分布）
    pub fn get_policy(&self) -> Vec<f32> {
        self.get_policy_with_temperature(1.0)
    }

    /// 获取搜索策略，应用温度参数
    /// temperature=1.0: 直接访问计数归一化
    /// temperature→0: 更集中在访问最多的走法
    /// temperature>1: 更均匀分布
    pub fn get_policy_with_temperature(&self, temperature: f32) -> Vec<f32> {
        let root = &self.nodes[self.root_index];
        let mut policy = vec![0.0; BOARD_WIDTH * BOARD_HEIGHT];

        if temperature == 0.0 {
            // 温度为0：one-hot 在访问最多的走法上
            let mut max_visits = 0;
            let mut best_move = None;
            for (&(x, y), &child_idx) in &root.children {
                let visits = self.nodes[child_idx].visit_count;
                if visits > max_visits {
                    max_visits = visits;
                    best_move = Some((x, y));
                }
            }
            if let Some((x, y)) = best_move {
                policy[x * BOARD_WIDTH + y] = 1.0;
            }
            return policy;
        }

        // 温度不为0：使用 visits^(1/T) 然后归一化
        let mut total = 0.0f32;
        let mut weighted_visits = std::collections::HashMap::new();

        for (&(x, y), &child_idx) in &root.children {
            let visits = self.nodes[child_idx].visit_count;
            let weight = (visits as f32).powf(1.0 / temperature);
            weighted_visits.insert((x, y), weight);
            total += weight;
        }

        if total > 0.0 {
            for ((x, y), weight) in weighted_visits {
                policy[x * BOARD_WIDTH + y] = weight / total;
            }
        }

        policy
    }

    /// 选择最佳走法
    pub fn select_move(&self, temperature: f32) -> (usize, usize) {
        let root = &self.nodes[self.root_index];

        if temperature == 0.0 {
            // 选择访问次数最多的；若无子节点，选择第一个空位
            let mut best_move: Option<(usize, usize)> = None;
            let mut max_visits = 0;

            for (&mv, &child_idx) in &root.children {
                let visits = self.nodes[child_idx].visit_count;
                if visits > max_visits {
                    max_visits = visits;
                    best_move = Some(mv);
                }
            }

            if let Some(mv) = best_move {
                return mv;
            }
            for i in 0..BOARD_HEIGHT {
                for j in 0..BOARD_WIDTH {
                    if self.states[self.root_index].get(i as i32, j as i32) == Some(0) {
                        return (i, j);
                    }
                }
            }
            return (0, 0);
        } else {
            // 根据访问次数的温度调整分布采样
            #[cfg(feature = "random")]
            {
                use rand::distributions::WeightedIndex;
                use rand::prelude::*;

                let moves: Vec<_> = root.children.keys().cloned().collect();
                let weights: Vec<_> = root
                    .children
                    .values()
                    .map(|&idx| (self.nodes[idx].visit_count as f32).powf(1.0 / temperature))
                    .collect();

                let dist = WeightedIndex::new(&weights).unwrap();
                let mut rng = thread_rng();
                moves[dist.sample(&mut rng)]
            }
            #[cfg(not(feature = "random"))]
            {
                // 如果没有随机特性，退化为贪婪选择
                self.select_move(0.0)
            }
        }
    }

    /// 将棋盘转换为张量
    fn board_to_tensor(&self, board: &Board, player: u8) -> Tensor {
        let mut tensor_data = vec![0.0f32; 3 * BOARD_WIDTH * BOARD_HEIGHT];

        for i in 0..BOARD_HEIGHT {
            for j in 0..BOARD_WIDTH {
                let cell = board.get(i as i32, j as i32);
                let idx = i * BOARD_WIDTH + j;

                match cell {
                    Some(p) if p == player => {
                        tensor_data[idx] = 1.0; // Channel 0: 当前玩家
                    }
                    Some(p) if p != 0 => {
                        tensor_data[BOARD_WIDTH * BOARD_HEIGHT + idx] = 1.0; // Channel 1: 对手
                    }
                    _ => {}
                }
            }
        }

        // Channel 2: 当前玩家标记
        let player_channel_start = 2 * BOARD_WIDTH * BOARD_HEIGHT;
        for i in player_channel_start..(3 * BOARD_WIDTH * BOARD_HEIGHT) {
            tensor_data[i] = if player == 1 { 1.0 } else { 0.0 };
        }

        Tensor::from_slice(&tensor_data)
            .view([1, 3, BOARD_HEIGHT as i64, BOARD_WIDTH as i64])
            .to_device(Device::cuda_if_available())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphazero_net::*;
    use tch::nn;

    #[test]
    #[cfg(feature = "alphazero")]
    fn test_alphazero_mcts() {
        let vs = nn::VarStore::new(Device::Cpu);
        let net = AlphaZeroNet::new(&vs.root(), 32, 2);

        let board = Board::new_default();
        let mut mcts = AlphaZeroMCTS::new(board, 10);

        mcts.search(&net, 1);
        let (x, y) = mcts.select_move(0.0);

        println!("✅ AlphaZero MCTS selected move: ({}, {})", x, y);
        assert!(x < 15 && y < 15);
    }
}
