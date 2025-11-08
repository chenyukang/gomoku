// AlphaZero 风格的 MCTS for Connect4

#![cfg(feature = "alphazero")]

use super::az_net::Connect4Net;
use super::connect4::Connect4;
use std::collections::HashMap;
use tch::Tensor;

const C_PUCT: f32 = 1.0; // UCB 公式中的探索常数

#[derive(Clone)]
struct Node {
    visit_count: u32,
    total_value: f32,
    prior: f32,
    children: HashMap<usize, Node>, // key = action (column)
}

impl Node {
    fn new(prior: f32) -> Self {
        Self {
            visit_count: 0,
            total_value: 0.0,
            prior,
            children: HashMap::new(),
        }
    }

    /// Q值：平均价值
    fn q(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f32
        }
    }

    /// U值：探索奖励
    fn u(&self, parent_visits: u32) -> f32 {
        C_PUCT * self.prior * (parent_visits as f32).sqrt() / (1.0 + self.visit_count as f32)
    }

    /// UCB分数：Q + U
    fn ucb(&self, parent_visits: u32) -> f32 {
        self.q() + self.u(parent_visits)
    }

    /// 选择最佳子节点
    fn select_child(&self) -> Option<usize> {
        if self.children.is_empty() {
            return None;
        }

        self.children
            .iter()
            .max_by(|(_, a), (_, b)| {
                let a_score = a.ucb(self.visit_count);
                let b_score = b.ucb(self.visit_count);
                a_score.partial_cmp(&b_score).unwrap()
            })
            .map(|(action, _)| *action)
    }
}

pub struct MCTS {
    root: Node,
    num_simulations: u32,
}

impl MCTS {
    pub fn new(num_simulations: u32) -> Self {
        Self {
            root: Node::new(1.0),
            num_simulations,
        }
    }

    /// 执行 MCTS 搜索
    /// 返回每个动作的访问次数分布
    pub fn search(&mut self, game: &Connect4, net: &Connect4Net) -> Vec<f32> {
        for _ in 0..self.num_simulations {
            let mut game_copy = game.clone();
            let _value = Self::simulate_recursive(&mut game_copy, &mut self.root, net);
        }

        // 计算访问次数分布
        self.get_action_probs()
    }

    /// 单次模拟（递归版本，改为关联函数）
    fn simulate_recursive(game: &mut Connect4, node: &mut Node, net: &Connect4Net) -> f32 {
        // 如果游戏结束，返回结果
        if game.is_game_over() {
            return match game.winner() {
                Some(0) => 0.0, // 平局
                Some(winner) => {
                    if winner == game.current_player() {
                        1.0 // 当前玩家赢
                    } else {
                        -1.0 // 当前玩家输
                    }
                }
                None => 0.0,
            };
        }

        // 如果节点未展开，展开并评估
        if node.children.is_empty() {
            return Self::expand_node(game, node, net);
        }

        // 选择子节点
        let action = node.select_child().unwrap();
        game.play(action).unwrap();

        // 递归搜索子节点（注意符号翻转）
        let value = if let Some(child) = node.children.get_mut(&action) {
            -Self::simulate_recursive(game, child, net)
        } else {
            0.0
        };

        // 回溯更新
        node.visit_count += 1;
        node.total_value += value;

        value
    }

    /// 展开节点并使用神经网络评估
    fn expand_node(game: &Connect4, node: &mut Node, net: &Connect4Net) -> f32 {
        let legal_moves = game.legal_moves();
        if legal_moves.is_empty() {
            return 0.0; // 平局
        }

        // 将棋盘转换为张量
        let board_vec = game.to_tensor();
        let board_tensor = Tensor::f_from_slice(&board_vec)
            .unwrap()
            .reshape(&[1, 3, 6, 7]);

        // 使用神经网络预测
        let (policy, value) = net.predict(&board_tensor);

        // 提取策略和价值
        // 将 tensor 转换为 Vec
        let mut policy_vec = vec![0.0f32; 7];
        policy.view([7]).copy_data(&mut policy_vec, 7);
        let value_scalar: f32 = value.double_value(&[]) as f32;

        // 为每个合法动作创建子节点
        for &action in &legal_moves {
            let prior = policy_vec[action].exp(); // log概率转为概率
            node.children.insert(action, Node::new(prior));
        }

        value_scalar
    }

    /// 获取动作概率分布（基于访问次数）
    fn get_action_probs(&self) -> Vec<f32> {
        let mut probs = vec![0.0; 7];
        let total_visits = self.root.visit_count as f32;

        if total_visits > 0.0 {
            for (action, child) in &self.root.children {
                probs[*action] = child.visit_count as f32 / total_visits;
            }
        }

        probs
    }

    /// 选择最佳动作
    /// temperature = 0: 选择访问次数最多的
    /// 根据访问次数选择动作
    pub fn select_action(&self, temperature: f32) -> usize {
        let probs = self.get_action_probs();

        // 只保留访问过的动作（有子节点的）
        let valid_actions: Vec<(usize, f32)> = self
            .root
            .children
            .iter()
            .map(|(&action, _)| (action, probs[action]))
            .collect();

        if valid_actions.is_empty() {
            panic!("没有可用的动作");
        }

        if temperature < 0.01 {
            // 确定性选择 - 选访问次数最多的
            valid_actions
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(action, _)| *action)
                .unwrap()
        } else {
            // 按温度采样
            use rand::distributions::{Distribution, WeightedIndex};
            let actions: Vec<usize> = valid_actions.iter().map(|(a, _)| *a).collect();
            let weights: Vec<f32> = valid_actions
                .iter()
                .map(|(_, p)| (p + 1e-8).powf(1.0 / temperature)) // 加一个小值避免全0
                .collect();

            // 检查是否所有权重都为0（或太小）
            let total_weight: f32 = weights.iter().sum();
            if total_weight < 1e-10 {
                // 如果所有权重都太小，使用最大概率的动作
                let max_idx = valid_actions
                    .iter()
                    .enumerate()
                    .max_by(|(_, (_, p1)), (_, (_, p2))| p1.partial_cmp(p2).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                actions[max_idx]
            } else {
                let dist = WeightedIndex::new(&weights).unwrap();
                let idx = dist.sample(&mut rand::thread_rng());
                actions[idx]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::az_net::Connect4Trainer;

    #[test]
    fn test_mcts_basic() {
        let game = Connect4::new();
        let trainer = Connect4Trainer::new(32, 0.001);
        let mut mcts = MCTS::new(50);

        let probs = mcts.search(&game, &trainer.net);

        // 检查概率分布
        assert_eq!(probs.len(), 7);
        let sum: f32 = probs.iter().sum();
        println!("概率总和: {}, 各列概率: {:?}", sum, probs);
        println!("根节点访问次数: {}", mcts.root.visit_count);

        // MCTS应该已经进行了一些搜索
        assert!(mcts.root.visit_count > 0);
        // 至少有些动作应该被探索
        assert!(probs.iter().any(|&p| p > 0.0));
    }

    #[test]
    fn test_action_selection() {
        let game = Connect4::new();
        let trainer = Connect4Trainer::new(32, 0.001);
        let mut mcts = MCTS::new(50);

        mcts.search(&game, &trainer.net);
        let action = mcts.select_action(0.0);

        assert!(action < 7); // 动作应该在合法范围内
    }
}
