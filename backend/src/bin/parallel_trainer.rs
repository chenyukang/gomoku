// 并行训练主程序
#![cfg(feature = "alphazero")]

use gomoku::alphazero_net::AlphaZeroTrainer;
use gomoku::parallel_training::TrainingDataPool;
use std::env;
use std::path::Path;
use std::process::{Child, Command};
use std::time::Instant;

struct ParallelTrainer {
    model_path: String,
    data_dir: String,
    num_workers: usize,
    games_per_worker: usize,
    iterations_per_round: usize,
}

impl ParallelTrainer {
    fn new(
        model_path: String,
        data_dir: String,
        num_workers: usize,
        games_per_worker: usize,
        iterations_per_round: usize,
    ) -> Self {
        Self {
            model_path,
            data_dir,
            num_workers,
            games_per_worker,
            iterations_per_round,
        }
    }

    /// 启动所有工作进程
    fn spawn_workers(&mut self) -> Vec<Child> {
        println!("Spawning {} workers...", self.num_workers);

        let mut processes = Vec::new();

        for worker_id in 0..self.num_workers {
            match Command::new("cargo")
                .args(&[
                    "run",
                    "--release",
                    "--bin",
                    "parallel_worker",
                    "--features",
                    "alphazero",
                    "--",
                    &worker_id.to_string(),
                    &self.model_path,
                    &self.data_dir,
                    &self.games_per_worker.to_string(),
                ])
                .spawn()
            {
                Ok(child) => {
                    println!("Worker {} started (pid: {})", worker_id, child.id());
                    processes.push(child);
                }
                Err(e) => {
                    eprintln!("Failed to spawn worker {}: {}", worker_id, e);
                }
            }
        }

        processes
    }

    /// 等待所有工作进程完成
    fn wait_for_workers(&mut self, processes: Vec<Child>) {
        println!("Waiting for all workers to complete...");

        for (worker_id, mut child) in processes.into_iter().enumerate() {
            match child.wait() {
                Ok(status) => {
                    println!("Worker {} finished with status: {}", worker_id, status);
                }
                Err(e) => {
                    eprintln!("Error waiting for worker {}: {}", worker_id, e);
                }
            }
        }
    }

    /// 收集所有工作进程的数据
    fn collect_training_data(&self) -> TrainingDataPool {
        let mut pool = TrainingDataPool::new(&self.data_dir, 100_000);
        let loaded = pool.load_from_all_workers(self.num_workers);

        println!("Collected {} training samples from all workers", loaded);
        pool
    }

    /// 执行一轮训练
    fn train_iteration(&self, trainer: &mut AlphaZeroTrainer, pool: &TrainingDataPool) {
        println!(
            "Training on {} samples for {} iterations...",
            pool.len(),
            self.iterations_per_round
        );

        let samples = pool.get_samples();
        trainer.train_on_samples(samples, self.iterations_per_round);

        // 保存模型
        trainer
            .save_model(&self.model_path)
            .expect("Failed to save model");
        println!("Model saved to {}", self.model_path);
    }

    /// 运行完整的训练循环
    fn run(&mut self, num_rounds: usize) {
        // 初始化或加载模型（from_file 现在会自动处理加载失败的情况）
        let mut trainer = AlphaZeroTrainer::from_file(&self.model_path, 0.001, 128, 10);

        for round in 0..num_rounds {
            println!("\n========== Round {}/{} ==========", round + 1, num_rounds);

            let round_start = Instant::now();

            // 1. 启动工作进程生成数据
            let processes = self.spawn_workers();

            // 2. 等待工作进程完成
            self.wait_for_workers(processes);

            // 3. 收集训练数据
            let pool = self.collect_training_data();

            if pool.len() == 0 {
                eprintln!("No training data collected in round {}", round + 1);
                continue;
            }

            // 4. 训练模型
            self.train_iteration(&mut trainer, &pool);

            let round_time = round_start.elapsed().as_secs_f32();
            println!(
                "Round {} completed in {:.1}s ({:.2} games/s)",
                round + 1,
                round_time,
                (self.num_workers * self.games_per_worker) as f32 / round_time
            );
        }

        println!("\nParallel training completed!");
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 6 {
        eprintln!(
            "Usage: {} <model_path> <data_dir> <num_workers> <games_per_worker> <rounds>",
            args[0]
        );
        eprintln!("Example: {} model.pt data 4 25 10", args[0]);
        std::process::exit(1);
    }

    let model_path = args[1].clone();
    let data_dir = args[2].clone();
    let num_workers: usize = args[3].parse().expect("Invalid num_workers");
    let games_per_worker: usize = args[4].parse().expect("Invalid games_per_worker");
    let num_rounds: usize = args[5].parse().expect("Invalid num_rounds");

    println!("Parallel Training Configuration:");
    println!("  Model path: {}", model_path);
    println!("  Data directory: {}", data_dir);
    println!("  Workers: {}", num_workers);
    println!("  Games per worker: {}", games_per_worker);
    println!("  Rounds: {}", num_rounds);
    println!(
        "  Total games per round: {}",
        num_workers * games_per_worker
    );

    let mut trainer = ParallelTrainer::new(
        model_path,
        data_dir,
        num_workers,
        games_per_worker,
        500, // iterations_per_round
    );

    trainer.run(num_rounds);
}
