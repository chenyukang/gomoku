#[cfg(feature = "alphazero")]
use crate::algo::GomokuSolver;
#[cfg(feature = "alphazero")]
use crate::az_resnet::Connect4ResNetTrainer;
#[cfg(feature = "alphazero")]
use crate::board::Move;
#[cfg(feature = "alphazero")]
use std::convert::TryFrom;
#[cfg(feature = "alphazero")]
use tch::{Device, Tensor};

#[cfg(feature = "alphazero")]
pub struct AlphaZero;

#[cfg(feature = "alphazero")]
static MODEL_PATH: &str = "connect4_resnet_converted.pt";

#[cfg(feature = "alphazero")]
impl AlphaZero {
    fn best_device() -> Device {
        if tch::utils::has_mps() {
            Device::Mps
        } else if tch::Cuda::is_available() {
            Device::cuda_if_available()
        } else {
            Device::Cpu
        }
    }

    fn load_model() -> Result<Connect4ResNetTrainer, Box<dyn std::error::Error>> {
        let device = Self::best_device();
        println!(
            "AlphaZero: Loading model from {} on {:?}",
            MODEL_PATH, device
        );

        // Create a trainer with the same architecture as training
        let mut trainer = Connect4ResNetTrainer::new(128, 10, 0.001);
        trainer.load(MODEL_PATH)?;

        Ok(trainer)
    }

    fn board_to_tensor(board_str: &str, width: usize, height: usize, current_player: u8) -> Tensor {
        let mut board = vec![vec![0i8; width]; height];

        for (i, ch) in board_str.chars().enumerate() {
            let row = i / width;
            let col = i % width;
            if row < height && col < width {
                board[row][col] = match ch {
                    '0' => 0,
                    '1' => 1,
                    '2' => 2,
                    _ => 0,
                };
            }
        }

        let device = Self::best_device();
        let mut current = vec![vec![0.0f32; width]; height];
        let mut opponent = vec![vec![0.0f32; width]; height];
        let mut empty = vec![vec![0.0f32; width]; height];

        let opponent_player = if current_player == 1 { 2 } else { 1 };

        for i in 0..height {
            for j in 0..width {
                match board[i][j] {
                    p if p == current_player as i8 => current[i][j] = 1.0,
                    p if p == opponent_player as i8 => opponent[i][j] = 1.0,
                    0 => empty[i][j] = 1.0,
                    _ => {}
                }
            }
        }

        let current_flat: Vec<f32> = current.iter().flatten().copied().collect();
        let opponent_flat: Vec<f32> = opponent.iter().flatten().copied().collect();
        let empty_flat: Vec<f32> = empty.iter().flatten().copied().collect();

        let current_tensor =
            Tensor::from_slice(&current_flat).view([1, 1, height as i64, width as i64]);
        let opponent_tensor =
            Tensor::from_slice(&opponent_flat).view([1, 1, height as i64, width as i64]);
        let empty_tensor =
            Tensor::from_slice(&empty_flat).view([1, 1, height as i64, width as i64]);

        Tensor::cat(&[current_tensor, opponent_tensor, empty_tensor], 1).to(device)
    }

    fn get_legal_moves(board_str: &str, width: usize, height: usize) -> Vec<usize> {
        let mut legal_cols = Vec::new();

        if height == 6 && width == 7 {
            // Connect4: 检查每列是否有空位（从底部往上）
            for col in 0..width {
                // 检查这一列的顶部（第0行）是否为空
                // 如果顶部为空，说明这一列还可以放棋子
                if board_str.chars().nth(col).unwrap() == '0' {
                    legal_cols.push(col);
                }
            }
            eprintln!("Connect4 legal columns: {:?}", legal_cols);
        } else {
            // Gomoku: 任何空位都可以下
            for (i, ch) in board_str.chars().enumerate() {
                if ch == '0' {
                    legal_cols.push(i);
                }
            }
            eprintln!("Gomoku legal positions: {} total", legal_cols.len());
        }

        legal_cols
    }
}

#[cfg(feature = "alphazero")]
impl GomokuSolver for AlphaZero {
    fn best_move(input: &str, width: usize, height: usize) -> Move {
        eprintln!(
            "AlphaZero: Deciding best move...: {} width: {} height: {}",
            input, width, height
        );
        let trainer = match Self::load_model() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("AlphaZero: Failed to load model: {}", e);
                return Move {
                    x: 0,
                    y: 0,
                    score: 0,
                    original_score: 0,
                };
            }
        };

        let count_1 = input.chars().filter(|&c| c == '1').count();
        let count_2 = input.chars().filter(|&c| c == '2').count();
        let current_player = if count_1 <= count_2 { 1 } else { 2 };

        let board_tensor = Self::board_to_tensor(input, width, height, current_player);

        // 调试：打印输入tensor的形状和部分内容
        eprintln!("Board tensor shape: {:?}", board_tensor.size());
        eprintln!("Current player: {}", current_player);
        
        let (policy_logits, value) = tch::no_grad(|| trainer.net.predict(&board_tensor));
        
        eprintln!("Value prediction: {:?}", value.double_value(&[]));

        // 重要：对 logits 应用 softmax 得到真正的概率分布
        let policy = policy_logits.softmax(-1, tch::Kind::Float);

        let legal_moves = Self::get_legal_moves(input, width, height);

        if legal_moves.is_empty() {
            return Move {
                x: 0,
                y: 0,
                score: 0,
                original_score: 0,
            };
        }

        let best_move = if height == 6 && width == 7 {
            // MPS 不支持 float64，直接使用 float32
            let policy_vec: Vec<f32> = Vec::try_from(policy.squeeze()).unwrap();
            
            eprintln!("Policy output (all 7 columns): {:?}", policy_vec);
            eprintln!("Legal columns: {:?}", legal_moves);

            let mut best_col = legal_moves[0];
            let mut best_prob = policy_vec[best_col];
            
            eprintln!("Initial: col={}, prob={}", best_col, best_prob);

            for &col in &legal_moves {
                if policy_vec[col] > best_prob {
                    eprintln!("Better move found: col={}, prob={} (was col={}, prob={})", 
                             col, policy_vec[col], best_col, best_prob);
                    best_prob = policy_vec[col];
                    best_col = col;
                }
            }

            let mut row = height - 1;
            for r in (0..height).rev() {
                if input.chars().nth(r * width + best_col).unwrap() == '0' {
                    row = r;
                    break;
                }
            }

            Move {
                x: row,
                y: best_col,
                score: (best_prob * 1000.0) as i32,
                original_score: (best_prob * 1000.0) as i32,
            }
        } else {
            // MPS 不支持 float64，直接使用 float32
            let policy_vec: Vec<f32> = Vec::try_from(policy.squeeze()).unwrap();

            let mut best_idx = legal_moves[0];
            let mut best_prob = policy_vec[best_idx];

            for &idx in &legal_moves {
                if idx < policy_vec.len() && policy_vec[idx] > best_prob {
                    best_prob = policy_vec[idx];
                    best_idx = idx;
                }
            }

            let row = best_idx / width;
            let col = best_idx % width;

            Move {
                x: row,
                y: col,
                score: (best_prob * 1000.0) as i32,
                original_score: (best_prob * 1000.0) as i32,
            }
        };

        println!(
            "AlphaZero: row={}, col={}, score={}",
            best_move.x, best_move.y, best_move.score
        );
        best_move
    }
}
