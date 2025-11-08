// AlphaZero 模型评估模块

#![cfg(feature = "alphazero")]

use super::az_mcts::MCTS;
use super::az_mcts_rollout::MCTSWithRollout;
use super::az_net::Connect4Net;
use super::connect4::Connect4;
use rand::Rng;
use std::collections::HashMap;

/// 玩家类型
pub enum Player<'a> {
    /// AlphaZero (神经网络 + MCTS)
    AlphaZero {
        net: &'a Connect4Net,
        simulations: u32,
    },
    /// AlphaZero with Rollout (训练初期使用)
    AlphaZeroRollout {
        net: &'a Connect4Net,
        simulations: u32,
    },
    /// 纯MCTS（无神经网络，使用随机rollout）
    PureMCTS { simulations: u32 },
    /// 随机玩家
    Random,
}

impl<'a> Player<'a> {
    /// 选择动作
    pub fn select_move(&self, game: &Connect4) -> usize {
        let legal_moves = game.legal_moves();
        if legal_moves.is_empty() {
            panic!("没有合法动作");
        }

        match self {
            Player::AlphaZero { net, simulations } => {
                let mut mcts = MCTS::new(*simulations);
                mcts.search(game, net);
                mcts.select_action(0.0) // 温度=0，选择最佳动作
            }
            Player::AlphaZeroRollout { net, simulations } => {
                let mut mcts = MCTSWithRollout::new(*simulations, true);
                mcts.search(game, net);
                mcts.select_action(0.0)
            }
            Player::PureMCTS { simulations } => {
                // 简单实现：对每个合法动作模拟N次
                let mut scores: HashMap<usize, f32> = HashMap::new();
                for &action in &legal_moves {
                    let mut total_score = 0.0;
                    for _ in 0..*simulations {
                        let mut sim_game = game.clone();
                        sim_game.play(action).ok();
                        total_score += Self::random_rollout(&mut sim_game, game.current_player());
                    }
                    scores.insert(action, total_score / *simulations as f32);
                }
                *scores
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            }
            Player::Random => {
                let mut rng = rand::thread_rng();
                legal_moves[rng.gen_range(0..legal_moves.len())]
            }
        }
    }

    /// 随机走到底，返回结果（从original_player视角）
    fn random_rollout(game: &mut Connect4, original_player: u8) -> f32 {
        let mut rng = rand::thread_rng();
        while !game.is_game_over() {
            let legal_moves = game.legal_moves();
            if legal_moves.is_empty() {
                break;
            }
            let action = legal_moves[rng.gen_range(0..legal_moves.len())];
            game.play(action).ok();
        }

        match game.winner() {
            Some(0) => 0.0,                         // 平局
            Some(w) if w == original_player => 1.0, // 赢
            Some(_) => -1.0,                        // 输
            None => 0.0,
        }
    }
}

/// 对战结果
#[derive(Debug, Clone)]
pub struct GameResult {
    pub winner: Option<u8>, // None=平局, Some(1)=玩家1赢, Some(2)=玩家2赢
    pub moves: usize,       // 总步数
}

/// 进行一局对战
pub fn play_game<'a>(player1: &Player<'a>, player2: &Player<'a>, verbose: bool) -> GameResult {
    let mut game = Connect4::new();
    let mut moves = 0;

    if verbose {
        println!("\n🎮 开始新游戏");
        game.print();
    }

    while !game.is_game_over() {
        let action = if game.current_player() == 1 {
            player1.select_move(&game)
        } else {
            player2.select_move(&game)
        };

        game.play(action).expect("非法动作");
        moves += 1;

        if verbose {
            println!("\n玩家 {} 落子第 {} 列", 3 - game.current_player(), action);
            game.print();
        }
    }

    if verbose {
        match game.winner() {
            Some(0) => println!("🤝 平局！"),
            Some(w) => println!("🏆 玩家 {} 获胜！", w),
            None => println!("游戏结束"),
        }
        println!("总步数: {}", moves);
    }

    GameResult {
        winner: game.winner(),
        moves,
    }
}

/// 评估统计
#[derive(Debug)]
pub struct EvalStats {
    pub player1_wins: usize,
    pub player2_wins: usize,
    pub draws: usize,
    pub total_games: usize,
    pub avg_moves: f32,
}

impl EvalStats {
    pub fn new() -> Self {
        Self {
            player1_wins: 0,
            player2_wins: 0,
            draws: 0,
            total_games: 0,
            avg_moves: 0.0,
        }
    }

    pub fn add_result(&mut self, result: &GameResult) {
        match result.winner {
            Some(1) => self.player1_wins += 1,
            Some(2) => self.player2_wins += 1,
            _ => self.draws += 1,
        }
        self.total_games += 1;
        self.avg_moves = (self.avg_moves * (self.total_games - 1) as f32 + result.moves as f32)
            / self.total_games as f32;
    }

    pub fn player1_winrate(&self) -> f32 {
        if self.total_games == 0 {
            return 0.0;
        }
        self.player1_wins as f32 / self.total_games as f32
    }

    pub fn player2_winrate(&self) -> f32 {
        if self.total_games == 0 {
            return 0.0;
        }
        self.player2_wins as f32 / self.total_games as f32
    }

    pub fn print(&self) {
        println!("\n📊 评估统计");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("总对局数: {}", self.total_games);
        println!(
            "玩家1: {} 胜 ({:.1}%)",
            self.player1_wins,
            self.player1_winrate() * 100.0
        );
        println!(
            "玩家2: {} 胜 ({:.1}%)",
            self.player2_wins,
            self.player2_winrate() * 100.0
        );
        println!(
            "平局:   {} ({:.1}%)",
            self.draws,
            self.draws as f32 / self.total_games as f32 * 100.0
        );
        println!("平均步数: {:.1}", self.avg_moves);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

/// 运行评估（多局对战）
pub fn evaluate<'a>(
    player1: &Player<'a>,
    player2: &Player<'a>,
    num_games: usize,
    verbose: bool,
) -> EvalStats {
    let mut stats = EvalStats::new();

    println!("\n🎯 开始评估: {} 局对战", num_games);

    for i in 0..num_games {
        if !verbose && (i + 1) % 10 == 0 {
            println!("  完成 {}/{} 局", i + 1, num_games);
        }

        let result = play_game(player1, player2, verbose);
        stats.add_result(&result);
    }

    stats.print();
    stats
}

/// 交换颜色再次评估（消除先手优势）
pub fn evaluate_symmetric<'a>(
    player1: &Player<'a>,
    player2: &Player<'a>,
    num_games_per_side: usize,
    verbose: bool,
) -> (EvalStats, EvalStats) {
    println!("\n🔄 对称评估（双方各执先后手）");

    println!("\n第一阶段: 玩家1先手");
    let stats1 = evaluate(player1, player2, num_games_per_side, verbose);

    println!("\n第二阶段: 玩家2先手");
    let stats2 = evaluate(player2, player1, num_games_per_side, verbose);

    println!("\n📈 综合统计:");
    let total_p1_wins = stats1.player1_wins + stats2.player2_wins;
    let total_p2_wins = stats1.player2_wins + stats2.player1_wins;
    let total_draws = stats1.draws + stats2.draws;
    let total_games = stats1.total_games + stats2.total_games;

    println!(
        "玩家1总胜场: {} ({:.1}%)",
        total_p1_wins,
        total_p1_wins as f32 / total_games as f32 * 100.0
    );
    println!(
        "玩家2总胜场: {} ({:.1}%)",
        total_p2_wins,
        total_p2_wins as f32 / total_games as f32 * 100.0
    );
    println!(
        "平局: {} ({:.1}%)",
        total_draws,
        total_draws as f32 / total_games as f32 * 100.0
    );

    (stats1, stats2)
}
