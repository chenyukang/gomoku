// MCTS with optional rollout - 训练初期用rollout而非网络value

#![cfg(feature = "alphazero")]

use super::az_net::Connect4Net;
use super::connect4::Connect4;
use std::collections::HashMap;
use tch::Tensor;

const C_PUCT: f32 = 1.0; // UCB 探索常数

#[derive(Clone)]
struct Node {
    visit_count: u32,
    total_value: f32,
    prior: f32,
    children: HashMap<usize, Node>,
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

    fn q(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f32
        }
    }

    fn u(&self, parent_visits: u32) -> f32 {
        C_PUCT * self.prior * (parent_visits as f32).sqrt() / (1.0 + self.visit_count as f32)
    }

    fn select_child(&self, legal_moves: &[usize]) -> Option<usize> {
        self.children
            .iter()
            .filter(|(action, _)| legal_moves.contains(action))
            .max_by(|(_, a), (_, b)| {
                let a_score = a.q() + a.u(self.visit_count);
                let b_score = b.q() + b.u(self.visit_count);
                a_score.partial_cmp(&b_score).unwrap()
            })
            .map(|(action, _)| *action)
    }
}

pub struct MCTSWithRollout {
    root: Node,
    num_simulations: u32,
    use_rollout: bool, // 是否使用rollout
}

impl MCTSWithRollout {
    pub fn new(num_simulations: u32, use_rollout: bool) -> Self {
        Self {
            root: Node::new(1.0),
            num_simulations,
            use_rollout,
        }
    }

    pub fn reset(&mut self) {
        self.root = Node::new(1.0);
    }

    pub fn search(&mut self, game: &Connect4, net: &Connect4Net) -> Vec<f32> {
        for _ in 0..self.num_simulations {
            let mut game_copy = game.clone();
            let _value = Self::simulate_recursive_static(
                &mut game_copy,
                &mut self.root,
                net,
                self.use_rollout,
            );
        }

        self.get_action_probs()
    }

    fn simulate_recursive_static(
        game: &mut Connect4,
        node: &mut Node,
        net: &Connect4Net,
        use_rollout: bool,
    ) -> f32 {
        if game.is_game_over() {
            return match game.winner() {
                Some(0) => 0.0,
                Some(winner) => {
                    if winner == game.current_player() {
                        1.0
                    } else {
                        -1.0
                    }
                }
                None => 0.0,
            };
        }

        if node.children.is_empty() {
            return Self::expand_node_static(game, node, net, use_rollout);
        }

        let legal_moves = game.legal_moves();
        let action = node.select_child(&legal_moves).unwrap();
        game.play(action).unwrap();

        let value = if let Some(child) = node.children.get_mut(&action) {
            -Self::simulate_recursive_static(game, child, net, use_rollout)
        } else {
            0.0
        };

        node.visit_count += 1;
        node.total_value += value;

        value
    }

    fn expand_node_static(
        game: &Connect4,
        node: &mut Node,
        net: &Connect4Net,
        use_rollout: bool,
    ) -> f32 {
        let legal_moves = game.legal_moves();
        if legal_moves.is_empty() {
            return 0.0;
        }

        // 关键改进：如果use_rollout，完全不用网络
        if use_rollout {
            // 均匀先验（不用网络）
            let uniform_prior = 1.0 / legal_moves.len() as f32;
            for &action in &legal_moves {
                node.children.insert(action, Node::new(uniform_prior));
            }
            
            // 用随机rollout评估
            return Self::random_rollout_static(game);
        }

        // 原来的逻辑：用神经网络
        let board_vec = game.to_tensor();
        let board_tensor = Tensor::f_from_slice(&board_vec)
            .unwrap()
            .reshape(&[1, 3, 6, 7]);

        let (policy, value) = net.predict(&board_tensor);

        let mut policy_vec = vec![0.0f32; 7];
        policy.view([7]).copy_data(&mut policy_vec, 7);

        let policy_sum: f32 = legal_moves.iter().map(|&m| policy_vec[m]).sum();
        let policy_sum = if policy_sum < 1e-8 { 1.0 } else { policy_sum };

        for &action in &legal_moves {
            let prior = policy_vec[action] / policy_sum;
            node.children.insert(action, Node::new(prior));
        }

        // 用神经网络value
        value.double_value(&[]) as f32
    }

    // 随机rollout到游戏结束
    fn random_rollout_static(game: &Connect4) -> f32 {
        let mut game_copy = game.clone();
        let original_player = game.current_player();

        use rand::Rng;
        let mut rng = rand::thread_rng();

        while !game_copy.is_game_over() {
            let legal_moves = game_copy.legal_moves();
            if legal_moves.is_empty() {
                break;
            }
            let action = legal_moves[rng.gen_range(0..legal_moves.len())];
            game_copy.play(action).ok();
        }

        match game_copy.winner() {
            Some(0) => 0.0,
            Some(w) if w == original_player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        }
    }

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

    pub fn select_action(&self, temperature: f32) -> usize {
        let probs = self.get_action_probs();
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
            valid_actions
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(action, _)| *action)
                .unwrap()
        } else {
            use rand::distributions::{Distribution, WeightedIndex};
            let actions: Vec<usize> = valid_actions.iter().map(|(a, _)| *a).collect();
            let weights: Vec<f32> = valid_actions
                .iter()
                .map(|(_, p)| (p + 1e-8).powf(1.0 / temperature))
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
