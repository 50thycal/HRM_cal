# HRM (Hierarchical Reasoning Model) Chat Interface Setup Guide

## 🎯 What I've Set Up For You

I've created an easy-to-use chat interface for your HRM (Hierarchical Reasoning Model) repository that allows you to interact with the AI model to solve puzzles like Sudoku, mazes, and ARC tasks.

## 📁 Files Created

### 1. **Demo Interface** (`demo_interface.py`) ✅ **READY TO USE**
- **Purpose**: Demonstrates how the chat interface works without requiring CUDA compilation
- **Usage**: `python3 demo_interface.py`
- **Features**: 
  - Interactive Sudoku puzzle input
  - Formatted output display
  - Help and example commands
  - Setup instructions for the real interface

### 2. **Full Interface** (`simple_chat.py`) ⚙️ **REQUIRES SETUP**
- **Purpose**: The actual chat interface that loads and runs the trained HRM model
- **Requirements**: CUDA toolkit, FlashAttention, and GPU
- **Features**:
  - Loads pre-trained checkpoints from HuggingFace
  - Real model inference
  - Interactive puzzle solving

### 3. **Original Interface** (`chat_interface.py`) 🔧 **ADVANCED**
- **Purpose**: More complex interface with full training pipeline integration
- **Note**: Requires additional dependencies and setup

## 🚀 Quick Start (Demo)

```bash
# Try the demo interface right now:
python3 demo_interface.py

# Commands you can use:
# - "example" - See an example Sudoku puzzle
# - "help" - Get help information
# - "real" - See setup instructions for the full interface
# - Enter a Sudoku puzzle (81 digits, 0 for empty cells)
# - "quit" - Exit the interface
```

## 🛠️ Full Setup (For Real AI Model)

### Prerequisites
- NVIDIA GPU with CUDA support
- Ubuntu/Linux system
- Python 3.8+

### Step 1: Install CUDA Toolkit
```bash
# Download and install CUDA 12.6
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run"
wget -O cuda_installer.run $CUDA_URL
sudo sh cuda_installer.run --silent --toolkit --override
export CUDA_HOME=/usr/local/cuda-12.6
```

### Step 2: Install Python Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install the project requirements
pip install -r requirements.txt

# Install FlashAttention (requires CUDA compilation)
pip install flash-attn

# Install additional build tools if needed
pip install packaging ninja wheel setuptools setuptools-scm
```

### Step 3: Download Pre-trained Models
```bash
# Download the Sudoku checkpoint (already done for you)
huggingface-cli download sapientinc/HRM-checkpoint-sudoku-extreme --local-dir checkpoints/hrm-sudoku-extreme

# Other available models:
# huggingface-cli download sapientinc/HRM-checkpoint-ARC-2 --local-dir checkpoints/hrm-arc-2
# huggingface-cli download sapientinc/HRM-checkpoint-maze-30x30-hard --local-dir checkpoints/hrm-maze
```

### Step 4: Run the Full Interface
```bash
# Run with default Sudoku checkpoint
python3 simple_chat.py

# Or specify a different checkpoint
python3 simple_chat.py --checkpoint checkpoints/hrm-arc-2/checkpoint
```

## 🧩 How to Use

### Sudoku Format
- Enter exactly 81 digits (9x9 grid flattened)
- Use `0` for empty cells
- Use digits `1-9` for filled cells
- Example: `003020600900305001001806400008102900700000008006708200002609500800203009005010300`

### Interactive Commands
- `help` - Show help information
- `example` - Display an example puzzle
- `quit` or `exit` - Exit the interface
- Any 81-digit string - Solve the Sudoku puzzle

## 🏆 Available Pre-trained Models

1. **Sudoku Extreme** (`hrm-sudoku-extreme`) ✅ **DOWNLOADED**
   - Trained on 1000 extremely difficult Sudoku puzzles
   - Near-perfect solving capability
   - ~27M parameters

2. **ARC-AGI-2** (`HRM-checkpoint-ARC-2`)
   - Trained on Abstraction and Reasoning Corpus tasks
   - Tests artificial general intelligence capabilities
   - Complex visual reasoning patterns

3. **Maze 30x30 Hard** (`HRM-checkpoint-maze-30x30-hard`)
   - Trained on large maze pathfinding problems
   - Optimal path finding in complex environments

## 🧠 How HRM Works

The Hierarchical Reasoning Model uses a novel architecture with:

- **High-level module**: Slow, abstract planning
- **Low-level module**: Rapid, detailed computations  
- **Sequential reasoning**: Processes in a single forward pass
- **No Chain-of-Thought**: Works without explicit step-by-step data
- **Small scale**: Only 27M parameters vs much larger models
- **High performance**: Near-perfect on challenging reasoning tasks

## 🐛 Troubleshooting

### Common Issues

1. **`ModuleNotFoundError: No module named 'flash_attn'`**
   - Solution: Install CUDA toolkit and compile FlashAttention
   - Alternative: Use the demo interface for testing

2. **`CUDA_HOME environment variable is not set`**
   - Solution: `export CUDA_HOME=/usr/local/cuda-12.6`
   - Add to your `.bashrc` for persistence

3. **GPU out of memory**
   - The model is designed to be efficient, but ensure you have sufficient GPU memory
   - Try reducing batch size in the configuration

4. **Model returns same puzzle**
   - This can happen if the model wasn't loaded correctly
   - Verify the checkpoint path and model compatibility

## 📚 Example Session

```
🎯 Welcome to the HRM Demo Interface!
==================================================
📋 This demonstrates how the real interface would work:
  • Sudoku: Enter 81 digits (0 for empty cells)
  • Type 'help' for more information
  • Type 'example' to see an example
  • Type 'quit' or 'exit' to stop
==================================================

🧩 Enter your Sudoku puzzle: example

📝 Example Sudoku Puzzle
=========================
Input: 003020600900305001001806400008102900700000008006708200002609500800203009005010300
Visual representation:
0 0 3 0 2 0 6 0 0
9 0 0 3 0 5 0 0 1
0 0 1 8 0 6 4 0 0
─────────────────
0 0 8 1 0 2 9 0 0
7 0 0 0 0 0 0 0 8
0 0 6 7 0 8 2 0 0
─────────────────
0 0 2 6 0 9 5 0 0
8 0 0 2 0 3 0 0 9
0 0 5 0 1 0 3 0 0

🧩 Enter your Sudoku puzzle: 003020600900305001001806400008102900700000008006708200002609500800203009005010300

🤔 Processing your puzzle... (demo mode)
🎯 Demo Solved Sudoku:
4 8 3 9 2 1 6 5 7
9 6 7 3 4 5 8 2 1
2 5 1 8 7 6 4 9 3
─────────────────
5 4 8 1 3 2 9 7 6
7 2 9 5 6 4 1 3 8
1 3 6 7 9 8 2 4 5
─────────────────
3 7 2 6 8 9 5 1 4
8 1 4 2 5 3 7 6 9
6 9 5 4 1 7 3 8 2
```

## 🎯 Next Steps

1. **Try the demo**: Run `python3 demo_interface.py` to see how it works
2. **Set up CUDA**: If you have an NVIDIA GPU, follow the full setup guide
3. **Experiment**: Try different Sudoku puzzles of varying difficulty
4. **Extend**: Modify the interface to support other puzzle types (mazes, ARC tasks)

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your CUDA and PyTorch installation
3. Make sure all dependencies are installed correctly
4. Try the demo interface first to understand the expected format

The HRM model represents a significant advancement in AI reasoning capabilities, and this interface makes it easy to interact with and test its capabilities!