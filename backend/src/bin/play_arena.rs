// AlphaZero vs 其他算法对弈工具
#![cfg(feature = "alphazero")]

use gomoku::alphazero_solver::AlphaZeroSolver;
use gomoku::board::Board;
use gomoku::minimax::MiniMax;
use gomoku::monte::MonteCarlo;
use gomoku::algo::GomokuSolver;
use std::io::{self, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    let model_path = &args[1];

    println!("🎮 AlphaZero Arena\n");
    println!("加载模型: {}", model_path);

    // 加载 AlphaZero 模型
    let alphazero = match AlphaZeroSolver::from_file(model_path, 32, 2, 100) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("❌ 加载模型失败: {}", e);
            return;
        }
    };

    println!("✅ 模型加载成功\n");

    // 选择对手
    println!("请选择对手:");
    println!("1. Monte Carlo (MCTS)");
    println!("2. Minimax (Alpha-Beta)");
    println!("3. 观看 AlphaZero 自我对弈");
    print!("\n请输入选择 (1-3): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let choice = input.trim();

    match choice {
        "1" => {
            println!("\n🎯 AlphaZero vs Monte Carlo");
            // pass simulation count; MonteCarlo will be constructed per move
            play_game(&alphazero, OpponentType::Monte(1000));
        }
        "2" => {
            println!("\n🎯 AlphaZero vs Minimax");
            // use Minimax solver via its trait implementation
            play_game(&alphazero, OpponentType::Minimax);
        }
        "3" => {
            println!("\n🎯 AlphaZero 自我对弈");
            self_play(&alphazero);
        }
        _ => {
            println!("❌ 无效选择");
        }
    }
}

enum OpponentType {
    Monte(u32), // simulation count
    Minimax,
}

fn play_game(alphazero: &AlphaZeroSolver, opponent: OpponentType) {
    let mut board = Board::new_default();
    let mut current_player = 1u8;
    let mut move_count = 0;

    println!("\n{}", "=".repeat(60));
    println!("游戏开始！");
    println!("玩家1 (●): AlphaZero");
    println!(
        "玩家2 (○): {}",
        match opponent {
                    OpponentType::Monte(_) => "Monte Carlo",
                    OpponentType::Minimax => "Minimax",
        }
    );
    println!("{}\n", "=".repeat(60));

    loop {
        move_count += 1;
        println!("\n--- 第 {} 步 ---", move_count);

        // 获取当前玩家的走法
        let next_move = if current_player == 1 {
            println!("🤖 AlphaZero 思考中...");
            alphazero.solve(&board, current_player)
        } else {
            println!(
                "🎲 {} 思考中...",
                match opponent {
                    OpponentType::Monte(_) => "Monte Carlo",
                    OpponentType::Minimax => "Minimax",
                }
            );
            match &opponent {
                OpponentType::Monte(sim_count) => {
                    let mut mc = MonteCarlo::new(board.clone(), current_player, *sim_count);
                    let mv = mc.search_move();
                    Some((mv.x as i32, mv.y as i32))
                }
                OpponentType::Minimax => {
                    let board_str = board.to_string();
                    let mv = MiniMax::best_move(&board_str);
                    Some((mv.x as i32, mv.y as i32))
                }
            }
        };

        if let Some((x, y)) = next_move {
            println!("走法: ({}, {})", x, y);
            board.place(x as usize, y as usize, current_player);

            // 打印棋盘
            print_board(&board, Some((x, y)));

            // 检查胜负
            if let Some(winner) = board.any_winner() {
                println!("\n{}", "=".repeat(60));
                if winner == 1 {
                    println!("🎉 AlphaZero 获胜!");
                } else if winner == 2 {
                    println!(
                        "🎉 {} 获胜!",
                        match opponent {
                            OpponentType::Monte(_) => "Monte Carlo",
                            OpponentType::Minimax => "Minimax",
                        }
                    );
                } else {
                    println!("🤝 平局!");
                }
                println!("{}", "=".repeat(60));
                println!("总步数: {}", move_count);
                break;
            }

            // 切换玩家
            current_player = if current_player == 1 { 2 } else { 1 };
        } else {
            println!("❌ 无效走法，游戏结束");
            break;
        }

        // 防止无限循环
        if move_count >= 225 {
            println!("\n🤝 棋盘已满，平局!");
            break;
        }
    }
}

fn self_play(alphazero: &AlphaZeroSolver) {
    let mut board = Board::new_default();
    let mut current_player = 1u8;
    let mut move_count = 0;

    println!("\n{}", "=".repeat(60));
    println!("AlphaZero 自我对弈");
    println!("{}\n", "=".repeat(60));

    loop {
        move_count += 1;
        println!("\n--- 第 {} 步 ---", move_count);

        println!("🤖 AlphaZero (玩家{}) 思考中...", current_player);
        let next_move = alphazero.solve(&board, current_player);

        if let Some((x, y)) = next_move {
            println!("走法: ({}, {})", x, y);
            board.place(x as usize, y as usize, current_player);

            // 打印棋盘
            print_board(&board, Some((x, y)));

            // 检查胜负
            if let Some(winner) = board.any_winner() {
                println!("\n{}", "=".repeat(60));
                if winner == 1 {
                    println!("🎉 玩家1 (●) 获胜!");
                } else if winner == 2 {
                    println!("🎉 玩家2 (○) 获胜!");
                } else {
                    println!("🤝 平局!");
                }
                println!("{}", "=".repeat(60));
                println!("总步数: {}", move_count);
                break;
            }

            // 切换玩家
            current_player = if current_player == 1 { 2 } else { 1 };
        } else {
            println!("❌ 无效走法，游戏结束");
            break;
        }

        if move_count >= 225 {
            println!("\n🤝 棋盘已满，平局!");
            break;
        }
    }
}

fn print_board(board: &Board, last_move: Option<(i32, i32)>) {
    println!("\n   0 1 2 3 4 5 6 7 8 9 A B C D E");
    for i in 0..15 {
        print!("{:2} ", i);
        for j in 0..15 {
            let cell = board.get(i as i32, j as i32);
            let is_last = if let Some((x, y)) = last_move {
                x == i as i32 && y == j as i32
            } else {
                false
            };

            match cell {
                Some(1) => {
                    if is_last {
                        print!("\x1b[31m●\x1b[0m "); // 红色标记最后一步
                    } else {
                        print!("● ");
                    }
                }
                Some(2) => {
                    if is_last {
                        print!("\x1b[31m○\x1b[0m "); // 红色标记最后一步
                    } else {
                        print!("○ ");
                    }
                }
                _ => print!(". "),
            }
        }
        println!();
    }
    println!();
}

fn print_usage() {
    println!("AlphaZero Arena - 对弈工具\n");
    println!("用法:");
    println!("  cargo run --release --features alphazero --bin play_arena -- <MODEL_PATH>\n");
    println!("示例:");
    println!(
        "  cargo run --release --features alphazero --bin play_arena -- ../data/test_model.pt\n"
    );
}
