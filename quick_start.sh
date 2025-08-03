#!/bin/bash
# Quick Start Script for HRM Chat Interface

echo "🚀 HRM (Hierarchical Reasoning Model) Quick Start"
echo "=================================================="
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Check if demo file exists
if [ ! -f "demo_interface.py" ]; then
    echo "❌ demo_interface.py not found. Please make sure you're in the correct directory."
    exit 1
fi

echo "🎯 Starting HRM Demo Interface..."
echo "   This demonstrates how the chat interface works."
echo "   You can enter Sudoku puzzles or use commands like 'example', 'help', etc."
echo ""
echo "💡 Tip: Type 'example' to see a sample Sudoku puzzle"
echo "💡 Tip: Type 'real' to see setup instructions for the full AI model"
echo "💡 Tip: Type 'quit' to exit"
echo ""

# Run the demo interface
python3 demo_interface.py