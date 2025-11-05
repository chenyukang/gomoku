#![cfg(feature = "alphazero")]
use super::alphazero_mcts::AlphaZeroMCTS;
use super::alphazero_net::AlphaZeroNet;
use super::board::Board;
use std::path::Path;
use tch::{nn, Device};

/// AlphaZero 求解器
pub struct AlphaZeroSolver {
    vs: nn::VarStore,
    net: AlphaZeroNet,
    num_simulations: u32,
    temperature: f32,
}

impl AlphaZeroSolver {
    /// 创建新的 AlphaZero 求解器
    pub fn new(num_filters: i64, num_res_blocks: i64, num_simulations: u32) -> Self {
        let vs = nn::VarStore::new(Device::Cpu);
        let net = AlphaZeroNet::new(&vs.root(), num_filters, num_res_blocks);

        Self {
            vs,
            net,
            num_simulations,
            temperature: 0.0, // 推理时使用确定性策略
        }
    }

    /// 从文件加载模型
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        num_filters: i64,
        num_res_blocks: i64,
        num_simulations: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut vs = nn::VarStore::new(Device::Cpu);
        let net = AlphaZeroNet::new(&vs.root(), num_filters, num_res_blocks);

        vs.load(path)?;

        Ok(Self {
            vs,
            net,
            num_simulations,
            temperature: 0.0,
        })
    }

    /// 设置温度参数
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }

    /// 获取神经网络的引用
    pub fn net(&self) -> &AlphaZeroNet {
        &self.net
    }

    /// 求解最佳走法
    pub fn solve(&self, board: &Board, player: u8) -> Option<(i32, i32)> {
        // 使用 MCTS 搜索最佳走法
        let mut mcts = AlphaZeroMCTS::new(board.clone(), self.num_simulations);
        mcts.search(&self.net, player);

        // 选择最佳走法
        let (x, y) = mcts.select_move(self.temperature);
        Some((x as i32, y as i32))
    }

    /// 显示求解器信息
    pub fn display(&self) -> String {
        format!(
            "AlphaZero (filters={}, simulations={})",
            self.net.num_filters(),
            self.num_simulations
        )
    }
}

/// 带自适应模拟次数的 AlphaZero 求解器
pub struct AdaptiveAlphaZeroSolver {
    solver: AlphaZeroSolver,
    min_simulations: u32,
    max_simulations: u32,
}

impl AdaptiveAlphaZeroSolver {
    pub fn new(
        num_filters: i64,
        num_res_blocks: i64,
        min_simulations: u32,
        max_simulations: u32,
    ) -> Self {
        let solver = AlphaZeroSolver::new(num_filters, num_res_blocks, max_simulations);

        Self {
            solver,
            min_simulations,
            max_simulations,
        }
    }

    pub fn from_file<P: AsRef<Path>>(
        path: P,
        num_filters: i64,
        num_res_blocks: i64,
        min_simulations: u32,
        max_simulations: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let solver =
            AlphaZeroSolver::from_file(path, num_filters, num_res_blocks, max_simulations)?;

        Ok(Self {
            solver,
            min_simulations,
            max_simulations,
        })
    }

    /// 根据棋盘状态调整模拟次数
    fn adjust_simulations(&self, board: &Board) -> u32 {
        let total_moves = board.total_moves();

        if total_moves < 10 {
            // 开局：使用较少的模拟
            self.min_simulations
        } else if total_moves < 30 {
            // 中局：逐渐增加模拟次数
            let progress = (total_moves - 10) as f32 / 20.0;
            let sims = self.min_simulations as f32
                + progress * (self.max_simulations - self.min_simulations) as f32;
            sims as u32
        } else {
            // 终局：使用最多的模拟
            self.max_simulations
        }
    }

    /// 求解最佳走法
    pub fn solve(&self, board: &Board, player: u8) -> Option<(i32, i32)> {
        let simulations = self.adjust_simulations(board);

        // 创建带自适应模拟次数的 MCTS
        let mut mcts = AlphaZeroMCTS::new(board.clone(), simulations);
        mcts.search(self.solver.net(), player);

        let (x, y) = mcts.select_move(self.solver.temperature);
        Some((x as i32, y as i32))
    }

    /// 显示求解器信息
    pub fn display(&self) -> String {
        format!(
            "AdaptiveAlphaZero (sims={}-{})",
            self.min_simulations, self.max_simulations
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "alphazero")]
    fn test_alphazero_solver() {
        let solver = AlphaZeroSolver::new(32, 2, 10);
        let mut board = Board::new_default();

        // 测试在空棋盘上走子
        if let Some((x, y)) = solver.solve(&board, 1) {
            println!("✅ AlphaZero move: ({}, {})", x, y);
            assert!(x >= 0 && x < 15);
            assert!(y >= 0 && y < 15);
        }
    }

    #[test]
    #[cfg(feature = "alphazero")]
    fn test_adaptive_solver() {
        let solver = AdaptiveAlphaZeroSolver::new(32, 2, 10, 50);
        let mut board = Board::new_default();

        // 测试自适应模拟次数
        board.place(7, 7, 1);
        board.place(7, 8, 2);

        if let Some((x, y)) = solver.solve(&board, 1) {
            println!("✅ Adaptive AlphaZero move: ({}, {})", x, y);
            assert!(x >= 0 && x < 15);
            assert!(y >= 0 && y < 15);
        }
    }
}
