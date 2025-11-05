// AlphaZero 训练管道
#![cfg(feature = "alphazero")]

use super::alphazero_mcts::AlphaZeroMCTS;
use super::alphazero_net::AlphaZeroTrainer;
use super::board::Board;
use super::utils::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tch::Tensor;

/// 训练样本
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub board: Vec<f32>,  // 3x15x15 的棋盘状态
    pub policy: Vec<f32>, // 225 的策略目标
    pub value: f32,       // 游戏结果 (-1, 0, 1)
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

        #[cfg(feature = "random")]
        {
            use rand::seq::SliceRandom;
            use rand::thread_rng;
            let mut rng = thread_rng();
            let samples: Vec<_> = self.samples.iter().collect();
            Some(
                samples
                    .choose_multiple(&mut rng, batch_size)
                    .map(|s| (*s).clone())
                    .collect(),
            )
        }

        #[cfg(not(feature = "random"))]
        {
            Some(self.samples.iter().take(batch_size).cloned().collect())
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }
}

/// AlphaZero 训练器
pub struct AlphaZeroPipeline {
    trainer: AlphaZeroTrainer,
    replay_buffer: ReplayBuffer,
    num_simulations: u32,
    temperature: f32,
    config: AlphaZeroConfig,
}

#[derive(Debug, Clone)]
pub struct AlphaZeroConfig {
    pub num_filters: i64,
    pub num_res_blocks: i64,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub replay_buffer_size: usize,
    pub num_self_play_games: usize,
    pub num_training_iterations: usize,
    pub num_mcts_simulations: u32,
    pub temperature: f32,
}

impl Default for AlphaZeroConfig {
    fn default() -> Self {
        Self {
            num_filters: 128,
            num_res_blocks: 10,
            learning_rate: 0.001,
            batch_size: 32,
            replay_buffer_size: 10000,
            num_self_play_games: 100,
            num_training_iterations: 1000,
            num_mcts_simulations: 400,
            temperature: 1.0,
        }
    }
}

impl AlphaZeroPipeline {
    pub fn new(config: AlphaZeroConfig) -> Self {
        // 设置 PyTorch 线程数以提升 CPU 利用率
        tch::set_num_threads(num_cpus::get() as i32);

        let trainer = AlphaZeroTrainer::new(
            config.num_filters,
            config.num_res_blocks,
            config.learning_rate,
        );
        let replay_buffer = ReplayBuffer::new(config.replay_buffer_size);

        Self {
            trainer,
            replay_buffer,
            num_simulations: config.num_mcts_simulations,
            temperature: config.temperature,
            config,
        }
    }

    /// 自我对弈生成训练数据
    pub fn self_play_game(&self) -> Vec<TrainingSample> {
        let mut samples = Vec::new();
        let mut board = Board::new_default();
        let mut current_player = 1u8;
        let mut game_states = Vec::new();

        // 游戏循环
        for step in 0..300 {
            // MCTS 搜索
            let mut mcts = AlphaZeroMCTS::new(board.clone(), self.num_simulations);
            mcts.search(self.trainer.net(), current_player);

            // 获取策略（使用温度参数生成训练目标）
            // 前30步使用 temperature=1.0 生成更多样化的目标
            // 后续使用 temperature=1.0 保持一致（也可以用更小的值让目标更尖锐）
            let policy_temperature = if step < 30 { 1.0 } else { 1.0 };
            let policy = mcts.get_policy_with_temperature(policy_temperature);

            // 记录状态和策略
            let board_tensor = self.board_to_vec(&board, current_player);
            game_states.push((board_tensor, policy, current_player));

            // 选择走法（使用温度调整采样）
            let move_temperature = if step < 30 { self.temperature } else { 0.0 };
            let (x, y) = mcts.select_move(move_temperature);

            // 执行走法
            board.place(x, y, current_player);

            // 检查游戏是否结束
            if let Some(winner) = board.any_winner() {
                // 分配奖励
                for (board_vec, policy, player) in game_states {
                    let value = if winner == player {
                        1.0
                    } else if winner == cfg::opponent(player) {
                        -1.0
                    } else {
                        0.0
                    };

                    samples.push(TrainingSample {
                        board: board_vec,
                        policy,
                        value,
                    });
                }
                break;
            }

            // 平局：没有空位了
            if board.empty_cells_count() == 0 {
                for (board_vec, policy, _player) in game_states {
                    samples.push(TrainingSample {
                        board: board_vec,
                        policy,
                        value: 0.0,
                    });
                }
                break;
            }

            current_player = cfg::opponent(current_player);
        }

        samples
    }

    /// 批量自我对弈
    pub fn generate_self_play_data(&mut self, num_games: usize) {
        println!("🎮 Generating {} self-play games...", num_games);

        let start = std::time::Instant::now();
        let mut total_samples = 0;

        for i in 0..num_games {
            if i % 5 == 0 && i > 0 {
                let elapsed = start.elapsed().as_secs_f32();
                let games_per_sec = i as f32 / elapsed;
                let remaining = (num_games - i) as f32 / games_per_sec;
                println!(
                    "  Progress: {}/{} ({:.1} games/s, ~{:.0}s remaining, {} samples)",
                    i, num_games, games_per_sec, remaining, total_samples
                );
            }

            let samples = self.self_play_game();
            total_samples += samples.len();
            self.replay_buffer.add_batch(samples);
        }

        let total_time = start.elapsed().as_secs_f32();
        println!(
            "✅ Generated {} training samples in {:.1}s ({:.1} games/s)",
            total_samples,
            total_time,
            num_games as f32 / total_time
        );
    }

    /// 训练网络（带验证集和早停）
    pub fn train(&mut self, num_iterations: usize) {
        println!("🎓 Training for {} iterations...", num_iterations);

        let mut best_loss = f64::INFINITY;
        let mut loss_history = Vec::new();
        let mut val_loss_history = Vec::new();
        let mut no_improvement_count = 0;
        let patience = 20; // 早停耐心值

        for iter in 0..num_iterations {
            // 训练批次
            if let Some(batch) = self.replay_buffer.sample_batch(self.config.batch_size) {
                let (boards, policies, values) = self.prepare_batch(&batch);
                let (total_loss, policy_loss, value_loss) =
                    self.trainer.train_batch(&boards, &policies, &values);

                loss_history.push(total_loss);

                // 每10次迭代计算验证损失
                if iter % 10 == 0 {
                    if let Some(val_batch) = self
                        .replay_buffer
                        .sample_batch(self.config.batch_size.min(self.replay_buffer.len() / 5))
                    {
                        let (val_boards, val_policies, val_values) = self.prepare_batch(&val_batch);
                        let val_loss =
                            self.trainer
                                .validate_batch(&val_boards, &val_policies, &val_values);
                        val_loss_history.push(val_loss);

                        // 早停检查
                        if val_loss < best_loss {
                            best_loss = val_loss;
                            no_improvement_count = 0;
                        } else {
                            no_improvement_count += 1;
                        }

                        if no_improvement_count >= patience {
                            println!("⚠️  Early stopping triggered at iteration {}", iter);
                            break;
                        }
                    }
                }

                if iter % 100 == 0 {
                    let recent_avg = if loss_history.len() >= 100 {
                        loss_history[loss_history.len() - 100..].iter().sum::<f64>() / 100.0
                    } else {
                        loss_history.iter().sum::<f64>() / loss_history.len() as f64
                    };

                    let val_info = if !val_loss_history.is_empty() {
                        format!(", Val={:.4}", val_loss_history.last().unwrap())
                    } else {
                        String::new()
                    };

                    println!(
                        "Iter {}/{}: Loss={:.4} (Policy={:.4}, Value={:.4}) | Avg={:.4}, Best={:.4}{}",
                        iter, num_iterations, total_loss, policy_loss, value_loss, recent_avg, best_loss, val_info
                    );
                }
            }
        }

        let final_avg = if loss_history.len() >= 100 {
            loss_history[loss_history.len() - 100..].iter().sum::<f64>() / 100.0
        } else {
            loss_history.iter().sum::<f64>() / loss_history.len() as f64
        };

        println!("✅ Training complete");
        println!("   Final avg loss (last 100): {:.4}", final_avg);
        println!("   Best val loss: {:.4}", best_loss);
    }

    /// 完整训练循环（改进版）
    pub fn train_loop(&mut self, num_iterations: usize) {
        println!("\n{}", "=".repeat(60));
        println!("🚀 AlphaZero Training Pipeline (Improved)");
        println!("{}\n", "=".repeat(60));

        println!("Configuration:");
        println!("  Iterations: {}", num_iterations);
        println!("  Games per iteration: {}", self.config.num_self_play_games);
        println!(
            "  Training iterations: {}",
            self.config.num_training_iterations
        );
        println!("  MCTS simulations: {}", self.config.num_mcts_simulations);
        println!("  Replay buffer size: {}", self.config.replay_buffer_size);
        println!("  Current buffer samples: {}", self.replay_buffer.len());
        println!();

        for iter in 0..num_iterations {
            println!("\n{}", "=".repeat(60));
            println!("📍 Iteration {}/{}", iter + 1, num_iterations);
            println!("{}", "=".repeat(60));

            // 🎮 自我对弈生成新数据（使用当前最新的网络）
            println!("\n🎮 Phase 1: Self-Play");
            self.generate_self_play_data(self.config.num_self_play_games);
            println!("   Buffer size: {} samples", self.replay_buffer.len());

            // 🎓 训练网络
            println!("\n🎓 Phase 2: Training");
            self.train(self.config.num_training_iterations);

            // 💾 保存检查点
            if (iter + 1) % 5 == 0 || iter == num_iterations - 1 {
                let path = format!("data/alphazero_model_iter_{}.pt", iter + 1);
                println!("\n💾 Saving checkpoint: {}", path);
                if let Err(e) = self.trainer.save(&path) {
                    eprintln!("   ⚠️  Failed to save: {}", e);
                } else {
                    println!("   ✅ Model saved");
                }
            }

            println!("\n✅ Iteration {} complete", iter + 1);
        }

        println!("\n{}", "=".repeat(60));
        println!("🎉 Training pipeline complete!");
        println!("   Total iterations: {}", num_iterations);
        println!("   Final buffer size: {} samples", self.replay_buffer.len());
        println!("{}", "=".repeat(60));
    }

    /// 将棋盘转换为向量
    fn board_to_vec(&self, board: &Board, player: u8) -> Vec<f32> {
        let flat = BOARD_WIDTH * BOARD_HEIGHT;
        let mut vec = vec![0.0; 3 * flat];

        for i in 0..BOARD_HEIGHT {
            for j in 0..BOARD_WIDTH {
                let cell = board.get(i as i32, j as i32);
                let idx = i * BOARD_WIDTH + j;

                match cell {
                    Some(p) if p == player => {
                        vec[idx] = 1.0;
                    }
                    Some(p) if p != 0 => {
                        vec[flat + idx] = 1.0;
                    }
                    _ => {}
                }
            }
        }

        for i in (2 * flat)..(3 * flat) {
            vec[i] = if player == 1 { 1.0 } else { 0.0 };
        }

        vec
    }

    /// 准备训练批次（带数据增强）
    fn prepare_batch(&self, samples: &[TrainingSample]) -> (Tensor, Tensor, Tensor) {
        let batch_size = samples.len();
        let flat = BOARD_WIDTH * BOARD_HEIGHT;

        let mut boards_data = Vec::with_capacity(batch_size * 3 * flat);
        let mut policies_data = Vec::with_capacity(batch_size * flat);
        let mut values_data = Vec::with_capacity(batch_size);

        #[cfg(feature = "random")]
        use rand::Rng;

        for sample in samples {
            // 数据增强：50% 概率水平翻转（适用于 Connect4）
            #[cfg(feature = "random")]
            let should_flip = rand::thread_rng().gen_bool(0.5);
            #[cfg(not(feature = "random"))]
            let should_flip = false;

            if should_flip {
                // 翻转棋盘（3 个通道都翻转）
                for ch in 0..3 {
                    for i in 0..BOARD_HEIGHT {
                        for j in 0..BOARD_WIDTH {
                            let dst_j = BOARD_WIDTH - 1 - j;
                            boards_data.push(sample.board[ch * flat + i * BOARD_WIDTH + dst_j]);
                        }
                    }
                }
                // 翻转策略
                for i in 0..BOARD_HEIGHT {
                    for j in 0..BOARD_WIDTH {
                        let dst_j = BOARD_WIDTH - 1 - j;
                        policies_data.push(sample.policy[i * BOARD_WIDTH + dst_j]);
                    }
                }
            } else {
                boards_data.extend_from_slice(&sample.board);
                policies_data.extend_from_slice(&sample.policy);
            }

            values_data.push(sample.value);
        }

        let boards = Tensor::from_slice(&boards_data).view([
            batch_size as i64,
            3,
            BOARD_HEIGHT as i64,
            BOARD_WIDTH as i64,
        ]);
        let policies = Tensor::from_slice(&policies_data).view([batch_size as i64, flat as i64]);
        let values = Tensor::from_slice(&values_data).view([batch_size as i64, 1]);

        (boards, policies, values)
    }

    /// 保存模型
    pub fn save_model(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.trainer.save(path)
    }

    /// 加载模型
    pub fn load_model(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.trainer.load(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "alphazero")]
    fn test_self_play() {
        let config = AlphaZeroConfig {
            num_filters: 32,
            num_res_blocks: 2,
            num_mcts_simulations: 10,
            ..Default::default()
        };

        let pipeline = AlphaZeroPipeline::new(config);
        let samples = pipeline.self_play_game();

        println!("✅ Self-play generated {} samples", samples.len());
        assert!(samples.len() > 0);
    }
}
