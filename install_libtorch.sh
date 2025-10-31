#!/bin/bash
# LibTorch 安装脚本（macOS）

set -e

echo "🚀 Installing LibTorch for tch-rs..."

# 检测操作系统
OS=$(uname -s)

if [ "$OS" = "Darwin" ]; then
    echo "📦 Detected macOS"

    # 检测芯片架构
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        echo "🍎 Apple Silicon detected"
        echo "⚠️  Note: tch-rs on Apple Silicon requires special setup"
        echo ""
        echo "Option 1: Use Python PyTorch installation"
        echo "  pip install torch torchvision"
        echo "  export LIBTORCH_USE_PYTORCH=1"
        echo ""
        echo "Option 2: Install via conda"
        echo "  conda install pytorch -c pytorch"
        echo "  export LIBTORCH=\$CONDA_PREFIX/lib/python3.x/site-packages/torch"
        echo ""
        read -p "Do you want to use Python PyTorch? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # 检查是否有 Python 和 pip
            if ! command -v pip3 &> /dev/null; then
                echo "❌ pip3 not found. Please install Python first."
                exit 1
            fi

            echo "📦 Installing PyTorch via pip..."
            pip3 install torch torchvision torchaudio

            # 获取 Python site-packages 路径
            TORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")

            echo ""
            echo "✅ PyTorch installed successfully!"
            echo ""
            echo "Add this to your ~/.zshrc or ~/.bash_profile:"
            echo "  export LIBTORCH_USE_PYTORCH=1"
            echo "  export DYLD_LIBRARY_PATH=\"$TORCH_PATH/lib:\$DYLD_LIBRARY_PATH\""
            echo ""
            echo "Then run: source ~/.zshrc"
        fi
    else
        echo "💻 Intel Mac detected"

        # 下载 CPU 版本的 LibTorch
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.1.0.zip"
        INSTALL_DIR="$HOME/libtorch"

        if [ -d "$INSTALL_DIR" ]; then
            echo "⚠️  $INSTALL_DIR already exists"
            read -p "Do you want to reinstall? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 0
            fi
            rm -rf "$INSTALL_DIR"
        fi

        echo "📥 Downloading LibTorch..."
        cd "$HOME"
        curl -L "$LIBTORCH_URL" -o libtorch.zip

        echo "📦 Extracting..."
        unzip -q libtorch.zip
        rm libtorch.zip

        echo ""
        echo "✅ LibTorch installed to: $INSTALL_DIR"
        echo ""
        echo "Add this to your ~/.zshrc or ~/.bash_profile:"
        echo "  export LIBTORCH=$INSTALL_DIR"
        echo "  export DYLD_LIBRARY_PATH=\"$INSTALL_DIR/lib:\$DYLD_LIBRARY_PATH\""
        echo ""
        echo "Then run: source ~/.zshrc"
    fi

elif [ "$OS" = "Linux" ]; then
    echo "🐧 Detected Linux"

    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
    INSTALL_DIR="$HOME/libtorch"

    if [ -d "$INSTALL_DIR" ]; then
        echo "⚠️  $INSTALL_DIR already exists"
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
        rm -rf "$INSTALL_DIR"
    fi

    echo "📥 Downloading LibTorch..."
    cd "$HOME"
    wget -q "$LIBTORCH_URL" -O libtorch.zip

    echo "📦 Extracting..."
    unzip -q libtorch.zip
    rm libtorch.zip

    echo ""
    echo "✅ LibTorch installed to: $INSTALL_DIR"
    echo ""
    echo "Add this to your ~/.bashrc:"
    echo "  export LIBTORCH=$INSTALL_DIR"
    echo "  export LD_LIBRARY_PATH=\"$INSTALL_DIR/lib:\$LD_LIBRARY_PATH\""
    echo ""
    echo "Then run: source ~/.bashrc"
else
    echo "❌ Unsupported OS: $OS"
    exit 1
fi

echo ""
echo "🎯 Next steps:"
echo "1. Update your shell configuration file with the export commands above"
echo "2. Reload your shell: source ~/.zshrc (or ~/.bashrc)"
echo "3. Build AlphaZero: cd backend && cargo build --features alphazero"
