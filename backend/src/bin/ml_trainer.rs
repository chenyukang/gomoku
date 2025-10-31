// ML 训练工具 - 用于生成训练数据
use gomoku::game_record::DatasetManager;
use gomoku::self_play::{SelfPlay, Tournament};

use clap::{App, Arg};

fn main() {
    let matches = App::new("Gomoku ML Trainer")
        .version("1.0")
        .author("Gomoku Team")
        .about("五子棋机器学习训练数据生成工具")
        .arg(
            Arg::new("selfplay")
                .long("selfplay")
                .takes_value(true)
                .help("自我对弈生成训练数据 (指定游戏数量)"),
        )
        .arg(
            Arg::new("tournament")
                .long("tournament")
                .takes_value(true)
                .help("锦标赛模式 (指定每对算法的游戏数量)"),
        )
        .arg(
            Arg::new("algo1")
                .long("algo1")
                .takes_value(true)
                .default_value("minimax")
                .help("算法1: minimax 或 monte_carlo"),
        )
        .arg(
            Arg::new("algo2")
                .long("algo2")
                .takes_value(true)
                .default_value("monte_carlo")
                .help("算法2: minimax 或 monte_carlo"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("详细输出"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .takes_value(true)
                .default_value("data/games")
                .help("输出文件前缀"),
        )
        .arg(
            Arg::new("exploration")
                .long("exploration")
                .takes_value(true)
                .help("启用随机开局模式 (指定前 N 步随机，最多3步，例如 --exploration 3)"),
        )
        .get_matches();

    // 自我对弈模式
    if let Some(num_games) = matches.value_of("selfplay") {
        let num: usize = num_games.parse().expect("请提供有效的游戏数量");
        let algo1 = matches.value_of("algo1").unwrap();
        let algo2 = matches.value_of("algo2").unwrap();
        let verbose = matches.is_present("verbose");
        let output_prefix = matches.value_of("output").unwrap();
        let exploration = matches.value_of("exploration");

        println!("🎮 自我对弈模式");
        println!("   游戏数量: {}", num);
        println!("   算法: {} vs {}", algo1, algo2);

        let self_play = if let Some(exp_str) = exploration {
            let opening_steps: usize = exp_str.parse().expect("随机开局步数必须是数字");
            println!("   🎲 随机开局: 前 {} 步随机", opening_steps.min(3));
            SelfPlay::new_with_random_opening(300, verbose, opening_steps)
        } else {
            SelfPlay::new(300, verbose)
        };

        println!("   输出: {}.json / {}.csv\n", output_prefix, output_prefix);

        let records = self_play.play_multiple_games(num, algo1, algo2);

        // 保存数据
        let mut dataset = DatasetManager::new();
        for record in records {
            dataset.add_game(record);
        }

        let json_file = format!("{}.json", output_prefix);
        let csv_file = format!("{}.csv", output_prefix);

        std::fs::create_dir_all("data").ok();

        dataset
            .save_dataset(&json_file, &csv_file)
            .expect("保存数据失败");

        println!("\n✅ 数据已保存到:");
        println!("   - {}", json_file);
        println!("   - {}", csv_file);

        dataset.print_stats();

        println!("\n💡 下一步:");
        println!("   python ml_examples/analyze_data.py");

        return;
    }

    // 锦标赛模式
    if let Some(games_per_pair) = matches.value_of("tournament") {
        let num: usize = games_per_pair.parse().expect("请提供有效的游戏数量");
        let output_prefix = matches.value_of("output").unwrap();

        let algorithms = vec!["minimax".to_string(), "monte_carlo".to_string()];

        let tournament = Tournament::new(algorithms, num);
        let records = tournament.run();

        // 保存数据
        let mut dataset = DatasetManager::new();
        for record in records {
            dataset.add_game(record);
        }

        let json_file = format!("{}_tournament.json", output_prefix);
        let csv_file = format!("{}_tournament.csv", output_prefix);

        std::fs::create_dir_all("data").ok();

        dataset
            .save_dataset(&json_file, &csv_file)
            .expect("保存数据失败");

        println!("\n✅ 锦标赛数据已保存");
        dataset.print_stats();

        return;
    }

    // 默认显示帮助
    println!("五子棋机器学习训练工具\n");
    println!("使用示例:");
    println!("  # 生成 10 局 minimax vs monte_carlo 的对局");
    println!("  cargo run --release --bin ml_trainer -- --selfplay 10");
    println!();
    println!("  # 使用随机开局增加多样性 (前3步随机)");
    println!("  cargo run --release --features random --bin ml_trainer -- --selfplay 100 --exploration 3");
    println!();
    println!("  # 生成 100 局数据用于训练");
    println!("  cargo run --release --bin ml_trainer -- --selfplay 100 --algo1 minimax --algo2 monte_carlo");
    println!();
    println!("  # 锦标赛模式: 所有算法互相对战");
    println!("  cargo run --release --bin ml_trainer -- --tournament 5");
    println!();
    println!("  # 详细模式 (显示棋盘)");
    println!("  cargo run --release --bin ml_trainer -- --selfplay 1 -v");
    println!();
    println!("注意:");
    println!("  - 使用 --exploration 需要编译时启用 random 特性");
    println!("  - 随机开局可以增加对局多样性，避免重复的棋局");
    println!();
    println!("更多帮助:");
    println!("  cargo run --bin ml_trainer -- --help");
}
