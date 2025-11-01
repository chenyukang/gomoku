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
        // 克隆当前状态以避免借用冲突
        let state = self.states[node_idx].clone();

        // 将棋盘转换为张量
        let board_tensor = self.board_to_tensor(&state, player);

        // 神经网络预测
        let (policy, value) = net.predict(&board_tensor);

        // 获取合法走法
        let mut board = state.clone();
        let valid_moves = board.gen_ordered_moves(player);

        if valid_moves.is_empty() {
            return value.double_value(&[]) as f32;
        }

        // 创建子节点
        // 将 Tensor 转换为 Vec<f32>
        let flat_size = BOARD_WIDTH * BOARD_HEIGHT;
        let policy_tensor = policy
            .view([flat_size as i64])
            .to_kind(Kind::Float)
            .to_device(Device::Cpu);
        let policy_data: Vec<f32> = (0..flat_size)
            .map(|i| policy_tensor.get(i as i64).double_value(&[]) as f32)
            .collect();

        for mv in valid_moves {
            let move_idx = mv.x * BOARD_WIDTH + mv.y;
            let prior = policy_data[move_idx as usize].exp(); // log_softmax -> softmax

            let child = AlphaNode::new(prior);
            let child_idx = self.nodes.len();
            self.nodes.push(child);

            // 创建子状态
            let mut child_state = state.clone();
            child_state.place(mv.x, mv.y, player);
            self.states.push(child_state);

            self.nodes[node_idx]
                .children
                .insert((mv.x, mv.y), child_idx);
        }

        value.double_value(&[]) as f32
    }

    /// 反向传播
    fn backpropagate(&mut self, path: &[usize], value: f32) {
        for &idx in path.iter().rev() {
            self.nodes[idx].visit_count += 1;
            self.nodes[idx].total_value += value;
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
        let root = &self.nodes[self.root_index];
        let mut policy = vec![0.0; BOARD_WIDTH * BOARD_HEIGHT];

        let total_visits: u32 = root
            .children
            .values()
            .map(|&idx| self.nodes[idx].visit_count)
            .sum();

        if total_visits == 0 {
            return policy;
        }

        for (&(x, y), &child_idx) in &root.children {
            let visits = self.nodes[child_idx].visit_count;
            policy[x * BOARD_WIDTH + y] = visits as f32 / total_visits as f32;
        }

        policy
    }

    /// 选择最佳走法
    pub fn select_move(&self, temperature: f32) -> (usize, usize) {
        let root = &self.nodes[self.root_index];

        if temperature == 0.0 {
            // 选择访问次数最多的
            let mut best_move = (7, 7);
            let mut max_visits = 0;

            for (&mv, &child_idx) in &root.children {
                let visits = self.nodes[child_idx].visit_count;
                if visits > max_visits {
                    max_visits = visits;
                    best_move = mv;
                }
            }

            best_move
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
