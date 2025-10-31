#!/bin/bash
# PyTorch 2.4.0 安装脚本（与 tch-rs 0.17 兼容）

set -e

echo "🚀 Installing PyTorch 2.4.0 for tch-rs 0.17..."

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python first."
    exit 1
fi

# 检查 pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Please install pip first."
    exit 1
fi

echo "📦 Current PyTorch version:"
python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null || echo "PyTorch not installed"

echo ""
echo "⚠️  Note: tch-rs 0.17 requires PyTorch 2.4.0"
echo ""
read -p "Do you want to install/downgrade to PyTorch 2.4.0? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

echo ""
echo "📥 Uninstalling current PyTorch..."
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo ""
echo "📦 Installing PyTorch 2.4.0..."

# 检测操作系统和架构
OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" = "Darwin" ]; then
    if [ "$ARCH" = "arm64" ]; then
        echo "🍎 Installing for Apple Silicon (macOS ARM64)..."
        # Apple Silicon 使用默认版本
        pip3 install torch==2.4.0 torchvision torchaudio
    else
        echo "💻 Installing for Intel Mac..."
        pip3 install torch==2.4.0 torchvision torchaudio
    fi
elif [ "$OS" = "Linux" ]; then
    echo "🐧 Installing for Linux..."
    pip3 install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "❌ Unsupported OS: $OS"
    exit 1
fi

# 验证安装
echo ""
echo "✅ Verifying installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 获取 PyTorch 路径
TORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")

echo ""
echo "✅ PyTorch 2.4.0 installed successfully!"
echo ""
echo "📝 Add these to your shell configuration (~/.zshrc or ~/.bashrc):"
echo ""
echo "  export LIBTORCH_USE_PYTORCH=1"

if [ "$OS" = "Darwin" ]; then
    echo "  export DYLD_LIBRARY_PATH=\"$TORCH_PATH/lib:\$DYLD_LIBRARY_PATH\""
    echo ""
    echo "Then run: source ~/.zshrc"
else
    echo "  export LD_LIBRARY_PATH=\"$TORCH_PATH/lib:\$LD_LIBRARY_PATH\""
    echo ""
    echo "Then run: source ~/.bashrc"
fi

echo ""
echo "🎯 Next steps:"
echo "1. Add the export commands to your shell config"
echo "2. Reload your shell configuration"
echo "3. Test: cd backend && cargo build --features alphazero --bin alphazero_cli"
