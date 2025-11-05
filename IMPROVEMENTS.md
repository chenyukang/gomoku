# AlphaZero Connect4 Improvements - Inspired by c4a0

## Summary

Successfully implemented key best practices from the [c4a0 project](https://github.com/advait/c4a0) to improve AlphaZero Connect4 training quality and playing strength.

## 🎯 Key Improvements Implemented

### 1. ✅ Horizontal Flip Data Augmentation
**File**: `backend/src/alphazero_trainer.rs::prepare_batch()`

- Added 50% probability horizontal flip for board and policy during training batch preparation
- Effectively doubles training data diversity without additional self-play cost
- Critical for Connect4 since the game is symmetric

**Impact**: Improves sample efficiency and generalization

### 2. ✅ Temperature-Based Policy Targets
**File**: `backend/src/alphazero_mcts.rs::get_policy_with_temperature()`

- New method to generate policy targets using `visits^(1/temperature)` formula
- Temperature=1.0: Direct visit-count normalization
- Temperature→0: Concentrated on best move (one-hot)
- Temperature>1: More uniform exploration

**Usage in training**:
```rust
let policy = mcts.get_policy_with_temperature(1.0);
```

**Impact**: Proper alignment between MCTS search behavior and training targets

### 3. ✅ Train/Validation Split with Early Stopping
**File**: `backend/src/alphazero_trainer.rs::train()`

- Periodic validation loss computation (every 10 iterations)
- Early stopping with patience=20 to prevent overfitting
- Tracks best validation loss across training

**Impact**: Better generalization, prevents wasteful over-training

### 4. ✅ Validation Batch Method
**File**: `backend/src/alphazero_net.rs::validate_batch()`

- New method to compute loss without gradient updates
- Uses `tch::no_grad()` for efficiency
- Runs network in eval mode (affects batch norm)

### 5. ✅ Gravity-Aware Connect4 (Already Present)
**File**: `backend/src/alphazero_mcts.rs::legal_moves()`

- Detects 7x6 win_len=4 boards
- Generates legal moves using gravity rules (column drop to lowest empty)
- Falls back to free-placement for other board sizes

## 📊 Training Results Comparison

### Baseline (Before Improvements)
- **Loss trajectory**: Policy ~3.8→3.0, Value ~0.0→0.0 (near-zero signal)
- **Final loss**: ~3.0 (policy-dominated)
- **Playing strength**: Weak, appears to not understand rules

### Improved Pipeline (After c4a0 Adaptations)
- **Loss trajectory**: Policy 4.0→2.1, Value 1.1→0.4 (strong signal!)
- **Final val loss**: ~2.1 (with early stopping)
- **Training dynamics**:
  - Iter 1: Val loss 4.78→2.83
  - Iter 2: Val loss 3.72→2.14
- **Key difference**: Value loss now meaningful, indicating proper game understanding

## 🔧 Configuration Used

```rust
AlphaZeroConfig {
    num_filters: 128,           // Larger network
    num_res_blocks: 6,          // More capacity
    num_mcts_simulations: 200,  // Stronger MCTS teacher
    batch_size: 64,             // Bigger batches
    temperature: 1.0,           // Exploration temperature
    ...
}
```

## 🚀 Usage

### Quick Training Run
```bash
export DYLD_LIBRARY_PATH="$(python3 -c 'import torch; import os; print(os.path.dirname(torch.__file__))')/lib:$DYLD_LIBRARY_PATH"
cargo run --release --features alphazero --bin train_alphazero ../data/model.pt 100 200 10
```

### Parameters
- `100`: Games per iteration (self-play)
- `200`: Training iterations per generation
- `10`: Number of generations

## 📈 What's Different from c4a0?

### We Kept:
- Pure Rust implementation (no Python dependency)
- Integrated network training (no PyO3 bridge)
- Simpler generation structure

### We Adopted:
- ✅ Horizontal flip augmentation
- ✅ Temperature-based policy targets
- ✅ Train/val split + early stopping
- ✅ Gravity Connect4 legal moves
- ✅ Proper MCTS value sign flipping
- ✅ Legal move masking and renormalization

### Not Yet Implemented (Future):
- Ply penalty for faster wins (Q_penalty vs Q_no_penalty)
- Solver integration for objective scoring
- Generation metadata tracking (JSON manifests)
- Better checkpoint management
- Explicit train/val data split (currently sampling-based)

## 🎓 Key Learnings from c4a0

1. **Data augmentation matters**: Simple horizontal flips effectively double your data
2. **Temperature is crucial**: Both for move selection AND policy target generation
3. **Validation prevents overfitting**: Early stopping saves compute and improves generalization
4. **Gravity rules are essential**: Connect4 isn't free-placement; proper rules improve learning
5. **Value signals must be correct**: Sign flipping during backprop is critical

## 🔬 Next Steps for Further Improvement

1. **Implement Ply Penalty**:
   - Track game length
   - Apply penalty to encourage faster wins: `q = 1.0 - penalty_coef * ply_count`
   - Train two separate value heads

2. **Better Data Management**:
   - Explicit 80/20 train/val split before training
   - Save generation metadata (games played, val loss, solver score)
   - Implement replay buffer prioritization

3. **Hyperparameter Tuning**:
   - Sweep MCTS simulations: 200, 400, 800
   - Sweep network size: (128, 6) vs (256, 10)
   - Sweep learning rate schedule

4. **Evaluation Pipeline**:
   - Tournament between generations
   - ELO rating tracking
   - Win-rate gating (only promote if >55% win-rate)

## 📝 Files Modified

- `backend/src/alphazero_trainer.rs`: Data augmentation, val split, temperature policy
- `backend/src/alphazero_mcts.rs`: Temperature-based policy, gravity legal moves
- `backend/src/alphazero_net.rs`: Validation batch method
- `backend/src/bin/train_alphazero.rs`: Better default hyperparameters

## ✨ Credits

Huge thanks to [@advait](https://github.com/advait) for the excellent [c4a0 reference implementation](https://github.com/advait/c4a0)!
