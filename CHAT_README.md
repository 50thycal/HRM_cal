# HRM Chat Interface 🤖

A simple and easy-to-use chat interface for the Hierarchical Reasoning Model (HRM). This interface allows you to interact with the AI model to solve various reasoning puzzles including Sudoku, mazes, and ARC (Abstraction and Reasoning Corpus) challenges.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Make the setup script executable
chmod +x setup_chat.sh

# Run the setup
./setup_chat.sh

# After setup, run any solver:
./run_sudoku.sh   # For Sudoku puzzles
./run_maze.sh     # For maze solving
./run_arc.sh      # For ARC puzzles
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the chat interface
python chat_with_hrm.py --model-type sudoku  # or maze, arc
```

## 📖 How to Use

### Sudoku Solver
1. Run: `python chat_with_hrm.py --model-type sudoku`
2. Enter your puzzle as 81 characters (use 0 or . for empty cells)
3. Example: `53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79`

### Maze Solver
1. Run: `python chat_with_hrm.py --model-type maze`
2. Draw your maze using:
   - `#` for walls
   - `S` for start position
   - `G` for goal position
   - ` ` (space) for empty paths

Example:
```
#########
#S      #
# ##### #
#     # #
##### # #
#     # #
# ##### #
#      G#
#########
```

### ARC Puzzle Solver
1. Run: `python chat_with_hrm.py --model-type arc`
2. Enter your puzzle as a grid of digits (0-9 represent different colors)
3. You can use space-separated format or JSON

## 💡 Commands

While in the chat interface:
- `help` - Show detailed instructions
- `example` - Show an example puzzle
- `quit` or `exit` - Exit the program

## 🎯 Features

- **Easy to use**: Simple command-line interface with clear instructions
- **Auto-download**: Automatically downloads pre-trained models on first run
- **Multiple puzzle types**: Supports Sudoku, mazes, and ARC puzzles
- **Visual output**: Formatted output with colors and symbols
- **Error handling**: Helpful error messages and recovery

## 🔧 Advanced Options

```bash
# Use a custom checkpoint
python chat_with_hrm.py --model-type sudoku --checkpoint path/to/model.pt

# Force CPU usage
python chat_with_hrm.py --model-type sudoku --device cpu
```

## 📝 Notes

- First run will download the model checkpoint (~100MB)
- GPU is used automatically if available (CUDA)
- Models are saved in the `checkpoints/` directory

## 🤝 Examples

The interface includes built-in examples for each puzzle type. Just type `example` when prompted to see them!

Enjoy solving puzzles with AI! 🧩✨