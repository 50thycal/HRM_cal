#!/bin/bash

# Setup script for HRM Chat Interface
echo "🚀 Setting up HRM Chat Interface..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install PyTorch (CPU version by default, modify for GPU)
echo "📥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected, installing CUDA-enabled PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi

# Install other requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Ask about optional web interface
echo ""
read -p "Would you like to install Gradio for web interface? [y/N]: " install_gradio
if [[ $install_gradio =~ ^[Yy]$ ]]; then
    echo "🌐 Installing Gradio..."
    pip install gradio
fi

# Create convenient launcher scripts
echo "✨ Creating launcher scripts..."

# Sudoku launcher
cat > run_sudoku.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python chat_with_hrm.py --model-type sudoku
EOF
chmod +x run_sudoku.sh

# Maze launcher
cat > run_maze.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python chat_with_hrm.py --model-type maze
EOF
chmod +x run_maze.sh

# ARC launcher
cat > run_arc.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python chat_with_hrm.py --model-type arc
EOF
chmod +x run_arc.sh

# Web interface launcher (if Gradio was installed)
if [[ $install_gradio =~ ^[Yy]$ ]]; then
cat > run_web.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python chat_web_interface.py
EOF
chmod +x run_web.sh
fi

echo "✅ Setup complete!"
echo ""
echo "🎯 Quick Start:"
echo "  - For Sudoku solver: ./run_sudoku.sh"
echo "  - For Maze solver:   ./run_maze.sh"
echo "  - For ARC solver:    ./run_arc.sh"
if [[ $install_gradio =~ ^[Yy]$ ]]; then
    echo "  - For Web Interface: ./run_web.sh"
fi
echo ""
echo "Or run directly with:"
echo "  python chat_with_hrm.py --model-type [sudoku|maze|arc]"
if [[ $install_gradio =~ ^[Yy]$ ]]; then
    echo "  python chat_web_interface.py  # For web interface"
fi
echo ""
echo "Note: The first run will download the model checkpoint (~100MB)"