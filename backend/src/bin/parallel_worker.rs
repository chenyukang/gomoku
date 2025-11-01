// 并行训练工作进程
#![cfg(feature = "alphazero")]

use gomoku::alphazero_mcts::AlphaZeroMCTS;
use gomoku::alphazero_net::AlphaZeroNet;
use gomoku::alphazero_trainer::TrainingSample;
use gomoku::board::Board;
use gomoku::parallel_training::WorkerDataWriter;
use std::env;
use std::time::Instant;
use tch::nn;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 5 {
        eprintln!(
            "Usage: {} <worker_id> <model_path> <output_dir> <num_games>",
            args[0]
        );
        std::process::exit(1);
    }

    let worker_id: usize = args[1].parse().expect("Invalid worker_id");
    let model_path = &args[2];
    let output_dir = &args[3];
    let num_games: usize = args[4].parse().expect("Invalid num_games");

    println!("Worker {} starting: {} games", worker_id, num_games);

    // 加载模型
    let device = tch::Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let net = AlphaZeroNet::new(&vs.root(), 128, 10);

    // 加载已训练的权重
    if std::path::Path::new(model_path).exists() {
        match vs.load(model_path) {
            Ok(_) => {
                println!("Worker {} loaded model from {}", worker_id, model_path);
            }
            Err(e) => {
                eprintln!("Worker {} failed to load model: {}", worker_id, e);
                eprintln!("Worker {} using fresh model instead", worker_id);
            }
        }
    } else {
        println!(
            "Worker {} using fresh model (no checkpoint found)",
            worker_id
        );
    }

    // 创建数据写入器
    let mut writer = WorkerDataWriter::new(worker_id, output_dir);

    // 生成自对弈数据
    let start = Instant::now();
    let mut total_samples = 0;
    let mut all_samples = Vec::new();

    for game_idx in 0..num_games {
        // 生成一局游戏
        let mut samples = self_play_game(&net);
        total_samples += samples.len();
        all_samples.append(&mut samples);

        // 每10局或最后一局写入一次
        if (game_idx + 1) % 10 == 0 || game_idx == num_games - 1 {
            if let Err(e) = writer.write_batch(&all_samples) {
                eprintln!("Worker {} failed to write batch: {}", worker_id, e);
            }
            all_samples.clear();
        }

        if (game_idx + 1) % 10 == 0 {
            let elapsed = start.elapsed().as_secs_f32();
            let games_per_sec = (game_idx + 1) as f32 / elapsed;
            println!(
                "Worker {} progress: {}/{} games ({:.2} games/s)",
                worker_id,
                game_idx + 1,
                num_games,
                games_per_sec
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    let games_per_sec = num_games as f32 / elapsed;

    println!(
        "Worker {} completed: {} games, {} samples in {:.1}s ({:.2} games/s)",
        worker_id, num_games, total_samples, elapsed, games_per_sec
    );
}

/// 自对弈生成一局游戏数据
fn self_play_game(net: &AlphaZeroNet) -> Vec<TrainingSample> {
    let mut board = Board::new_default(); // 使用默认构造函数
    let mut samples = Vec::new();
    let mut player = 1u8; // 1 = 黑方, 2 = 白方

    let mut move_count = 0;

    loop {
        // MCTS搜索
        let mut mcts = AlphaZeroMCTS::new(board.clone(), 100); // 100次模拟
        mcts.search(net, player);
        let policy = mcts.get_policy();

        // 温度参数：前30步用1.0，后面用0.0
        let temperature = if move_count < 30 { 1.0 } else { 0.0 };

        // 选择动作
        let (x, y) = mcts.select_move(temperature);

        // 将棋盘转换为向量表示
        let board_vec = board_to_vec(&board, player);

        // 保存样本
        samples.push(TrainingSample {
            board: board_vec,
            policy: policy,
            value: 0.0, // 临时值，游戏结束后会更新
        });

        // 执行动作
        board.place(x, y, player);
        move_count += 1;

        // 检查游戏是否结束
        if let Some(winner) = board.any_winner() {
            // 更新所有样本的价值
            update_sample_values(&mut samples, winner);
            break;
        }

        // 检查是否平局（简化：超过100步）
        if move_count >= 100 {
            update_sample_values_draw(&mut samples);
            break;
        }

        player = if player == 1 { 2 } else { 1 };
    }

    samples
}

/// 将棋盘转换为 3×15×15 的向量表示
fn board_to_vec(board: &Board, player: u8) -> Vec<f32> {
    let mut vec = Vec::with_capacity(3 * 15 * 15);

    // Channel 0: 当前玩家的棋子
    for i in 0..15 {
        for j in 0..15 {
            let cell = board.get(i as i32, j as i32);
            let value = if cell == Some(player) { 1.0 } else { 0.0 };
            vec.push(value);
        }
    }

    // Channel 1: 对手的棋子
    let opponent = if player == 1 { 2 } else { 1 };
    for i in 0..15 {
        for j in 0..15 {
            let cell = board.get(i as i32, j as i32);
            let value = if cell == Some(opponent) { 1.0 } else { 0.0 };
            vec.push(value);
        }
    }

    // Channel 2: 当前玩家标记（全1或全0）
    for _ in 0..15 * 15 {
        vec.push(if player == 1 { 1.0 } else { 0.0 });
    }

    vec
}

/// 更新样本的价值（根据游戏结果）
fn update_sample_values(samples: &mut [TrainingSample], winner: u8) {
    // 根据游戏结果更新所有样本的价值
    // 奇数步是黑方（player 1），偶数步是白方（player 2）
    for (idx, sample) in samples.iter_mut().enumerate() {
        let sample_player = if idx % 2 == 0 { 1u8 } else { 2u8 };

        if winner == sample_player {
            sample.value = 1.0; // 该样本的玩家赢了
        } else {
            sample.value = -1.0; // 该样本的玩家输了
        }
    }
}

/// 更新样本的价值（平局）
fn update_sample_values_draw(samples: &mut [TrainingSample]) {
    for sample in samples.iter_mut() {
        sample.value = 0.0;
    }
}
