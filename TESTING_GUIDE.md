# 🧪 HRM Testing Guide

This guide walks you through testing all features of the Hierarchical Reasoning Model (HRM) chat interface.

## 📋 Prerequisites

Dependencies have been installed. If you encounter any issues, run:
```bash
pip install --break-system-packages torch einops tqdm coolname pydantic argdantic huggingface_hub adam-atan2 numpy pyyaml gradio
```

## 🚀 Quick Test Commands

### 1. Test Sudoku Solver
```bash
# Interactive mode
./run_sudoku.sh

# Or directly
python3 chat_with_hrm.py --model-type sudoku
```

**Test puzzle:** `53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79`

### 2. Test Maze Solver
```bash
# Interactive mode
./run_maze.sh

# Or directly
python3 chat_with_hrm.py --model-type maze
```

**Test maze:**
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

### 3. Test ARC Solver
```bash
# Interactive mode
./run_arc.sh

# Or directly
python3 chat_with_hrm.py --model-type arc
```

**Test input:** `1 0 0 0 1 0 0 0 1` or `[[1,0,0],[0,1,0],[0,0,1]]`

### 4. Test Web Interface
```bash
# Launch web UI
./run_web.sh

# Or directly
python3 chat_web_interface.py
```
Then open http://localhost:7860 in your browser.

## 📝 Interactive Commands

When in any solver:
- `help` - Show detailed instructions
- `example` - Show an example puzzle
- `quit` or `exit` - Exit the program

## 🎯 What to Test

1. **Model Loading**: First run will download models (~100MB each)
2. **Input Validation**: Try invalid inputs to see error handling
3. **Solution Quality**: Verify the AI solves puzzles correctly
4. **Performance**: Note solving time (GPU vs CPU)

## 🔍 Demo Scripts

Run these to see examples:
```bash
python3 test_sudoku_demo.py
python3 test_maze_demo.py
python3 test_arc_demo.py
python3 test_web_interface.py
```

## ⚠️ Common Issues

1. **Missing dependencies**: Install with pip command above
2. **Model download fails**: Check internet connection
3. **Out of memory**: Use CPU mode with `--device cpu`
4. **Permission errors**: Use `chmod +x *.sh` for scripts

## 💡 Tips

- Models are cached in `checkpoints/` directory
- First run is slower due to model download
- GPU (CUDA) is automatically used if available
- Web interface supports all three puzzle types

Happy testing! 🧩✨