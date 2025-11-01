// 并行训练数据管理模块
#![cfg(feature = "alphazero")]

use super::alphazero_trainer::TrainingSample;
use std::fs;
use std::path::{Path, PathBuf};

/// 训练数据池 - 管理多个工作进程生成的数据
pub struct TrainingDataPool {
    data_dir: PathBuf,
    samples: Vec<TrainingSample>,
    max_samples: usize,
}

impl TrainingDataPool {
    pub fn new<P: AsRef<Path>>(data_dir: P, max_samples: usize) -> Self {
        let data_dir = data_dir.as_ref().to_path_buf();
        fs::create_dir_all(&data_dir).ok();

        Self {
            data_dir,
            samples: Vec::new(),
            max_samples,
        }
    }

    /// 从工作进程的数据文件加载样本
    pub fn load_from_worker(&mut self, worker_id: usize) -> std::io::Result<usize> {
        let worker_dir = self.data_dir.join(format!("worker_{}", worker_id));
        let mut loaded = 0;

        if !worker_dir.exists() {
            return Ok(0);
        }

        // 读取所有 JSON 文件
        for entry in fs::read_dir(&worker_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(data) = fs::read_to_string(&path) {
                    if let Ok(batch) = serde_json::from_str::<Vec<TrainingSample>>(&data) {
                        self.add_batch(batch.clone());
                        loaded += batch.len();

                        // 删除已读取的文件
                        fs::remove_file(&path).ok();
                    }
                }
            }
        }

        Ok(loaded)
    }

    /// 从所有工作进程加载数据
    pub fn load_from_all_workers(&mut self, num_workers: usize) -> usize {
        let mut total = 0;
        for worker_id in 0..num_workers {
            if let Ok(count) = self.load_from_worker(worker_id) {
                total += count;
            }
        }
        total
    }

    /// 添加样本批次
    pub fn add_batch(&mut self, mut batch: Vec<TrainingSample>) {
        self.samples.append(&mut batch);

        // 保持最大样本数限制
        if self.samples.len() > self.max_samples {
            let excess = self.samples.len() - self.max_samples;
            self.samples.drain(0..excess);
        }
    }

    /// 获取所有样本
    pub fn get_samples(&self) -> &[TrainingSample] {
        &self.samples
    }

    /// 样本数量
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// 清空数据
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// 随机采样
    #[cfg(feature = "random")]
    pub fn sample_batch(&self, batch_size: usize) -> Option<Vec<TrainingSample>> {
        use rand::seq::SliceRandom;

        if self.samples.len() < batch_size {
            return None;
        }

        let mut rng = rand::thread_rng();
        let batch = self
            .samples
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect();

        Some(batch)
    }

    #[cfg(not(feature = "random"))]
    pub fn sample_batch(&self, batch_size: usize) -> Option<Vec<TrainingSample>> {
        if self.samples.len() < batch_size {
            return None;
        }
        Some(self.samples[..batch_size].to_vec())
    }
}

/// 工作进程数据写入器
pub struct WorkerDataWriter {
    worker_id: usize,
    output_dir: PathBuf,
    batch_counter: usize,
}

impl WorkerDataWriter {
    pub fn new<P: AsRef<Path>>(worker_id: usize, output_dir: P) -> Self {
        let output_dir = output_dir.as_ref().join(format!("worker_{}", worker_id));
        fs::create_dir_all(&output_dir).ok();

        Self {
            worker_id,
            output_dir,
            batch_counter: 0,
        }
    }

    /// 写入一批训练样本
    pub fn write_batch(&mut self, samples: &[TrainingSample]) -> std::io::Result<()> {
        let filename = format!("batch_{:06}.json", self.batch_counter);
        let filepath = self.output_dir.join(&filename);

        let json = serde_json::to_string(samples)?;
        fs::write(&filepath, json)?;

        self.batch_counter += 1;
        println!(
            "Worker {} wrote {} samples to {}",
            self.worker_id,
            samples.len(),
            filename
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_pool() {
        let pool = TrainingDataPool::new("test_data", 1000);
        assert_eq!(pool.len(), 0);
    }
}
