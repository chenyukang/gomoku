// AlphaZero 训练器 - 自对弈和训练逻辑

#![cfg(feature = "alphazero")]

use super::az_mcts_rollout::MCTSWithRollout;
use super::az_net::{Connect4Net, Connect4Trainer};
use super::connect4::Connect4;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::convert::TryFrom;
use tch::Tensor;

/// 训练样本
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub board: Vec<f32>,  // (3, 6, 7) flatten后的棋盘
    pub policy: Vec<f32>, // (7,) MCTS搜索得到的策略
    pub outcome: f32,     // 游戏结果: 1=当前玩家赢, -1=输, 0=平局
}

/// 回放缓冲区
pub struct ReplayBuffer {
    samples: VecDeque<TrainingSample>,
    max_size: usize,
}

impl ReplayBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    pub fn add(&mut self, sample: TrainingSample) {
        if self.samples.len() >= self.max_size {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    pub fn add_batch(&mut self, samples: Vec<TrainingSample>) {
        for sample in samples {
            self.add(sample);
        }
    }

    pub fn sample_batch(&self, batch_size: usize) -> Option<Vec<TrainingSample>> {
        if self.samples.len() < batch_size {
            return None;
        }

        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let samples_vec: Vec<_> = self.samples.iter().cloned().collect();
        Some(
            samples_vec
                .choose_multiple(&mut rng, batch_size)
                .cloned()
                .collect(),
        )
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }
}

pub struct AlphaZeroTrainer {
    pub trainer: Connect4Trainer,
    replay_buffer: ReplayBuffer,
    num_mcts_simulations: u32,
}

impl AlphaZeroTrainer {
    pub fn new(
        num_filters: i64,
        learning_rate: f64,
        replay_buffer_size: usize,
        num_mcts_simulations: u32,
    ) -> Self {
        Self {
            trainer: Connect4Trainer::new(num_filters, learning_rate),
            replay_buffer: ReplayBuffer::new(replay_buffer_size),
            num_mcts_simulations,
        }
    }

    /// 进行一局自对弈
    pub fn self_play_game(&mut self, temperature: f32) -> Vec<TrainingSample> {
        let mut game = Connect4::new();
        let mut history = Vec::new();

        // 游戏循环
        while !game.is_game_over() {
            // 使用MCTS搜索 - 关键改进：使用rollout而非网络value
            let mut mcts = MCTSWithRollout::new(self.num_mcts_simulations, true);
            let policy = mcts.search(&game, &self.trainer.net);

            // 保存当前状态
            history.push((game.to_tensor(), policy.clone(), game.current_player()));

            // 选择动作（使用温度）
            let action = mcts.select_action(temperature);

            // 执行动作
            game.play(action).expect("非法动作");
        }

        // 根据游戏结果生成训练样本
        self.create_training_samples(history, game.winner())
    }

    /// 根据游戏结果创建训练样本
    fn create_training_samples(
        &self,
        history: Vec<(Vec<f32>, Vec<f32>, u8)>,
        winner: Option<u8>,
    ) -> Vec<TrainingSample> {
        let mut samples = Vec::new();

        for (board, policy, player) in history {
            // 计算该样本的结果
            let outcome = match winner {
                Some(0) => 0.0,                // 平局
                Some(w) if w == player => 1.0, // 该玩家赢了
                Some(_) => -1.0,               // 该玩家输了
                None => 0.0,                   // 不应该发生
            };

            samples.push(TrainingSample {
                board,
                policy,
                outcome,
            });
        }

        samples
    }

    /// 生成多局自对弈数据
    pub fn generate_self_play_data(&mut self, num_games: usize, temperature: f32) {
        println!("🎮 生成 {} 局自对弈数据...", num_games);

        for i in 0..num_games {
            let samples = self.self_play_game(temperature);
            self.replay_buffer.add_batch(samples);

            if (i + 1) % 10 == 0 {
                println!("  完成 {}/{} 局", i + 1, num_games);
            }
        }

        println!("✅ 自对弈完成，缓冲区大小: {}", self.replay_buffer.len());
    }

    /// 训练网络
    pub fn train(&mut self, batch_size: usize, num_iterations: usize) {
        println!(
            "🎯 开始训练，批次大小: {}, 迭代次数: {}",
            batch_size, num_iterations
        );

        let mut total_loss = 0.0;
        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;

        for i in 0..num_iterations {
            if let Some(batch) = self.replay_buffer.sample_batch(batch_size) {
                let (boards, policies, values) = self.prepare_batch(batch);
                let (loss, policy_loss, value_loss) =
                    self.trainer.train_batch(&boards, &policies, &values);

                total_loss += loss;
                total_policy_loss += policy_loss;
                total_value_loss += value_loss;

                if (i + 1) % 20 == 0 {
                    println!(
                        "  迭代 {}/{}: loss={:.4} (policy={:.4}, value={:.4})",
                        i + 1,
                        num_iterations,
                        loss,
                        policy_loss,
                        value_loss
                    );
                }
            } else {
                println!("⚠️  缓冲区样本不足，跳过训练");
                break;
            }
        }

        let avg_loss = total_loss / num_iterations as f64;
        let avg_policy_loss = total_policy_loss / num_iterations as f64;
        let avg_value_loss = total_value_loss / num_iterations as f64;

        println!(
            "✅ 训练完成，平均损失: {:.4} (policy={:.4}, value={:.4})",
            avg_loss, avg_policy_loss, avg_value_loss
        );
    }

    /// 准备训练批次
    fn prepare_batch(&self, batch: Vec<TrainingSample>) -> (Tensor, Tensor, Tensor) {
        let batch_size = batch.len();

        // 提取数据
        let boards: Vec<f32> = batch.iter().flat_map(|s| s.board.iter().cloned()).collect();

        let policies: Vec<f32> = batch
            .iter()
            .flat_map(|s| s.policy.iter().cloned())
            .collect();

        let values: Vec<f32> = batch.iter().map(|s| s.outcome).collect();

        // 转换为张量
        let boards_tensor =
            Tensor::f_from_slice(&boards)
                .unwrap()
                .reshape(&[batch_size as i64, 3, 6, 7]);

        let policies_tensor = Tensor::f_from_slice(&policies)
            .unwrap()
            .reshape(&[batch_size as i64, 7]);

        let values_tensor = Tensor::f_from_slice(&values)
            .unwrap()
            .reshape(&[batch_size as i64, 1]);

        (boards_tensor, policies_tensor, values_tensor)
    }

    /// 保存模型
    pub fn save_model(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.trainer.save(path)
    }

    /// 加载模型
    pub fn load_model(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.trainer.load(path)
    }

    /// 直接添加训练样本到replay buffer
    pub fn add_sample(&mut self, sample: TrainingSample) {
        self.replay_buffer.add(sample);
    }

    /// 获取replay buffer大小
    pub fn replay_buffer_size(&self) -> usize {
        self.replay_buffer.len()
    }

    /// 获取网络引用（用于MCTS）
    pub fn get_net(&self) -> &Connect4Net {
        &self.trainer.net
    }

    /// 使用网络预测一个局面
    pub fn predict(&self, game: &Connect4) -> (Tensor, f32) {
        let device = self.trainer.device();
        let board_tensor = Tensor::from_slice(&game.to_tensor())
            .view([1, 3, 6, 7])
            .to_device(device);
        
        let (policy_logits, value) = self.trainer.net.forward(&board_tensor, false);
        
        // 获取合法动作
        let valid_moves = game.legal_moves();
        let mut mask = vec![-1e9; 7];
        for &m in &valid_moves {
            mask[m] = 0.0;
        }
        let mask_tensor = Tensor::from_slice(&mask)
            .view([1, 7])
            .to_device(device);
        
        // 应用mask并softmax
        let policy = (policy_logits + mask_tensor).softmax(1, tch::Kind::Float);
        
        // 直接从value tensor获取标量值，避免MPS float64问题
        // 使用内部方法直接读取，不经过类型转换
        let value_scalar = unsafe {
            let value_data = value.data_ptr() as *const f32;
            *value_data
        };
        
        (policy.squeeze(), value_scalar)
    }
}
