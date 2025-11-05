# AlphaZero 模型加载问题修复说明

## 问题描述

用户发现 `train_iterative.sh` 脚本中只是复制了父模型，但 `train_alphazero` 程序并没有加载它。

经过检查，我们发现：
1. ✅ `train_alphazero.rs` 已添加模型加载代码
2. ❌ 模型加载时出现 PyTorch 格式兼容性错误
3. 💡 需要修复加载机制或使用替代方案

## 错误信息

```
Internal torch error: isGenericDict() INTERNAL ASSERT FAILED
Expected GenericDict but got Object
```

这是 `tch-rs` 与某些 PyTorch 版本的兼容性问题。

## 解决方案

### 方案 A：单次运行多个 Epochs（推荐）

不使用 `train_iterative.sh`，直接运行更多 epochs：

```bash
cd backend

# 训练50个epochs（相当于50代）
# 每个epoch：100局游戏 + 300次训练迭代
cargo run --release --features alphazero --bin train_alphazero \
    ../data/model.pt \
    100 \    # 每轮游戏数
    300 \    # 每轮训练迭代
    50       # 总轮数

# 训练完成后转换模型
cd ..
python3 convert_model.py data/model.pt data/model_converted.pt

# 测试
./play_match.sh data/model_converted.pt 10 500
```

**优点**：
- 不需要模型加载，避免兼容性问题
- 经验缓冲区在内存中累积，数据利用率高
- 一次性完成训练

**缺点**：
- 不能中途停止后继续
- 内存占用可能较大（但10万条缓冲区应该问题不大）

### 方案 B：修复模型保存/加载格式

需要修改 `alphazero_net.rs` 使用更兼容的保存格式。

**选项 1：使用 safetensors 格式**
```rust
// 需要添加依赖
// Cargo.toml: safetensors = "0.3"

pub fn save(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // 使用 safetensors 格式
    self.vs.save_to_safetensors(path)?;
    Ok(())
}

pub fn load(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    self.vs.load_from_safetensors(path)?;
    Ok(())
}
```

**选项 2：使用pickle格式（需要Python）**
```rust
// 保存为 pickle 格式，然后用 convert_model.py 转换
```

**选项 3：检查 tch-rs 版本**
可能是 tch-rs 版本太旧或太新，尝试更新/降级。

### 方案 C：手动管理检查点

每隔N个epochs手动保存并重启：

```bash
# 第1-10代
cargo run --release --features alphazero --bin train_alphazero ../data/gen_0010.pt 100 300 10

# 手动转换和备份
python3 convert_model.py data/gen_0010.pt data/gen_0010_converted.pt
cp data/gen_0010_converted.pt data/checkpoints/

# 第11-20代（从头开始，但可以用转换后的模型初始化）
cargo run --release --features alphazero --bin train_alphazero ../data/gen_0020.pt 100 300 10
```

## 推荐实践

### 初学者/快速测试
```bash
# 10个epochs，约1小时
cd backend
cargo run --release --features alphazero --bin train_alphazero ../data/quick.pt 100 300 10
cd ..
python3 convert_model.py data/quick.pt data/quick_converted.pt
./play_match.sh data/quick_converted.pt 5 500
```

### 标准训练
```bash
# 50个epochs，约5-10小时（可以后台运行）
cd backend
nohup cargo run --release --features alphazero --bin train_alphazero \
    ../data/standard.pt 100 300 50 > ../training.log 2>&1 &

# 监控进度
tail -f ../training.log

# 训练完成后转换
python3 convert_model.py data/standard.pt data/standard_converted.pt
```

### 深度训练（分批进行）
```bash
# 第1批：50 epochs
cargo run --release --features alphazero --bin train_alphazero ../data/batch1.pt 100 500 50

# 测试棋力
python3 convert_model.py data/batch1.pt data/batch1_converted.pt
./play_match.sh data/batch1_converted.pt 5 500

# 如果效果好，继续第2批：50 epochs
cargo run --release --features alphazero --bin train_alphazero ../data/batch2.pt 150 600 50
```

## 待修复

1. **模型加载兼容性**
   - 调查 tch-rs 保存格式
   - 测试不同的序列化方法
   - 可能需要自定义保存/加载逻辑

2. **迭代训练脚本**
   - 暂时禁用 `train_iterative.sh`（或标记为实验性）
   - 提供明确的错误信息
   - 添加模型格式检查

3. **文档更新**
   - 说明当前限制
   - 提供工作流程示例
   - 更新 QUICKSTART 和 TRAINING_GUIDE

## 参考

- `backend/src/bin/train_alphazero.rs` - 训练程序
- `backend/src/alphazero_trainer.rs` - 训练逻辑
- `backend/src/alphazero_net.rs` - 网络和保存/加载
- `convert_model.py` - 模型转换脚本
- tch-rs documentation: https://docs.rs/tch/latest/tch/
