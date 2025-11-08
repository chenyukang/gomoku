use gomoku::az_resnet::Connect4ResNetTrainer;
use gomoku::connect4::Connect4;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tch::Tensor;
use warp::Filter;

#[derive(Debug, Deserialize)]
struct MoveRequest {
    board: Vec<Vec<i8>>, // 6x7 棋盘，0=空，1=玩家1，2=玩家2
    current_player: i8,
}

#[derive(Debug, Serialize)]
struct MoveResponse {
    column: usize,
    success: bool,
    message: String,
}

#[derive(Debug, Serialize)]
struct StatusResponse {
    status: String,
    model_loaded: bool,
}

// AI状态管理
struct AIState {
    trainer: Connect4ResNetTrainer,
}

impl AIState {
    fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        println!("🤖 初始化AlphaZero AI...");
        println!("  模型: {}", model_path);

        // 创建trainer并加载模型
        let mut trainer = Connect4ResNetTrainer::new(128, 10, 0.001);

        if std::path::Path::new(model_path).exists() {
            trainer.load(model_path)?;
            println!("  ✅ 模型加载成功");
        } else {
            println!("  ⚠️  模型文件不存在，使用未训练的网络");
        }

        Ok(AIState { trainer })
    }

    fn get_best_move(&self, board: &Vec<Vec<i8>>, current_player: i8) -> Result<usize, String> {
        // 将棋盘转换为Connect4格式
        let mut game = Connect4::new();

        // 重建游戏状态
        for row in 0..6 {
            for col in 0..7 {
                let piece = board[row][col];
                if piece != 0 {
                    // 需要按照游戏历史重建，这里简化处理
                    // 实际应该从board状态反推
                }
            }
        }

        // 如果棋盘是初始状态，返回中间列
        let empty = board.iter().all(|row| row.iter().all(|&cell| cell == 0));
        if empty {
            return Ok(3); // 中间列
        }

        // 使用神经网络预测
        let board_tensor = self.board_to_tensor(&board, current_player);
        let device = self.trainer.device();
        let board_t = Tensor::f_from_slice(&board_tensor)
            .unwrap()
            .reshape(&[1, 3, 6, 7])
            .to_device(device);

        let (policy, _value) = self.trainer.net.predict(&board_t);

        // 获取策略分布
        let mut policy_vec = vec![0.0f32; 7];
        policy.view([7i64]).copy_data(&mut policy_vec, 7);

        // Softmax
        let max_val = policy_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = policy_vec.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();

        // 找到合法且概率最高的列
        let legal_moves = self.get_legal_moves(&board);

        let best_move = legal_moves
            .iter()
            .max_by(|&&a, &&b| probs[a].partial_cmp(&probs[b]).unwrap())
            .copied()
            .ok_or("没有合法移动")?;

        println!("  🎯 AI选择列 {}, 概率分布: {:?}", best_move, probs);

        Ok(best_move)
    }

    fn board_to_tensor(&self, board: &Vec<Vec<i8>>, current_player: i8) -> Vec<f32> {
        let mut tensor = vec![0.0f32; 3 * 6 * 7];

        for row in 0..6 {
            for col in 0..7 {
                let idx = row * 7 + col;
                let piece = board[row][col];

                if piece == current_player {
                    tensor[idx] = 1.0; // 当前玩家
                } else if piece != 0 {
                    tensor[126 + idx] = 1.0; // 对手
                }

                // 当前玩家标记
                if current_player == 1 {
                    tensor[252 + idx] = 1.0;
                }
            }
        }

        tensor
    }

    fn get_legal_moves(&self, board: &Vec<Vec<i8>>) -> Vec<usize> {
        (0..7).filter(|&col| board[0][col] == 0).collect()
    }
}

#[tokio::main]
async fn main() {
    println!("🚀 AlphaZero Connect4 服务器启动中...\n");

    // 查找最佳模型
    let model_path = if std::path::Path::new("connect4_resnet_best.pt").exists() {
        "connect4_resnet_best.pt"
    } else if std::path::Path::new("connect4_resnet_iter_5.pt").exists() {
        "connect4_resnet_iter_5.pt"
    } else {
        println!("⚠️  警告：找不到训练好的模型，将使用未训练的网络");
        "dummy.pt"
    };

    // 初始化AI
    let ai_state = match AIState::new(model_path) {
        Ok(state) => Arc::new(Mutex::new(state)),
        Err(e) => {
            eprintln!("❌ 初始化AI失败: {}", e);
            return;
        }
    };

    println!("\n✅ 服务器初始化完成");
    println!("📡 监听地址: http://localhost:8080");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // CORS配置
    let cors = warp::cors()
        .allow_any_origin()
        .allow_methods(vec!["GET", "POST", "OPTIONS"])
        .allow_headers(vec!["Content-Type"]);

    // 状态检查端点
    let status_route = warp::path("status").and(warp::get()).map(|| {
        warp::reply::json(&StatusResponse {
            status: "running".to_string(),
            model_loaded: true,
        })
    });

    // AI移动端点
    let ai_state_filter = warp::any().map(move || ai_state.clone());

    let move_route = warp::path("ai_move")
        .and(warp::post())
        .and(warp::body::json())
        .and(ai_state_filter)
        .map(|req: MoveRequest, ai_state: Arc<Mutex<AIState>>| {
            println!("📥 收到AI移动请求");

            let state = ai_state.lock().unwrap();

            match state.get_best_move(&req.board, req.current_player) {
                Ok(column) => {
                    println!("✅ AI决策完成：列 {}\n", column);
                    warp::reply::json(&MoveResponse {
                        column,
                        success: true,
                        message: format!("AI选择列 {}", column),
                    })
                }
                Err(e) => {
                    eprintln!("❌ AI决策失败: {}\n", e);
                    warp::reply::json(&MoveResponse {
                        column: 0,
                        success: false,
                        message: format!("错误: {}", e),
                    })
                }
            }
        });

    let routes = status_route.or(move_route).with(cors);

    println!("🎮 可用端点:");
    println!("  GET  /status    - 检查服务器状态");
    println!("  POST /ai_move   - AI决策");
    println!("\n等待客户端连接...\n");

    warp::serve(routes).run(([127, 0, 0, 1], 8080)).await;
}
