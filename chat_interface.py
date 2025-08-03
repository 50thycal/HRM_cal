#!/usr/bin/env python3
"""
Simple Chat Interface for Hierarchical Reasoning Model (HRM)

This script provides an easy way to interact with the trained HRM model
for solving various types of puzzles including Sudoku, mazes, and ARC tasks.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import argparse

# Add the current directory to the Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pretrain import PretrainConfig, init_train_state
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from dataset.common import PuzzleDatasetMetadata
from utils.functions import load_model_class


class HRMChatInterface:
    def __init__(self, checkpoint_path: str):
        """Initialize the chat interface with a pre-trained model."""
        self.checkpoint_path = checkpoint_path
        self.config = None
        self.model = None
        self.train_state = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🧠 Initializing HRM Chat Interface...")
        print(f"📱 Device: {self.device}")
        
        self._load_model()
        
    def _load_model(self):
        """Load the pre-trained model and configuration."""
        # Load configuration
        config_path = os.path.join(os.path.dirname(self.checkpoint_path), "all_config.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        self.config = PretrainConfig(**config_dict)
        
        # Create dummy metadata for model initialization (based on Sudoku format)
        dummy_metadata = PuzzleDatasetMetadata(
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            vocab_size=11,  # PAD + "0" ... "9" (encoded as 1-10)
            seq_len=81,  # 9x9 Sudoku grid
            num_puzzle_identifiers=1,
            total_groups=1,
            mean_puzzle_examples=1.0,
            sets=["all"]
        )
        
        # Initialize model
        self.train_state = init_train_state(self.config, dummy_metadata, world_size=1)
        
        # Load checkpoint
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.train_state.model.load_state_dict(checkpoint, assign=True)
        except Exception as e:
            # Try removing torch.compile prefix if it exists
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            filtered_checkpoint = {k.removeprefix("_orig_mod."): v for k, v in checkpoint.items()}
            self.train_state.model.load_state_dict(filtered_checkpoint, assign=True)
        
        self.train_state.model.eval()
        print(f"✅ Model loaded successfully from {self.checkpoint_path}")
        
    def solve_sudoku(self, sudoku_string: str) -> str:
        """
        Solve a Sudoku puzzle from a string representation.
        
        Args:
            sudoku_string: String representation of Sudoku (81 characters, 0 for empty cells)
        
        Returns:
            String representation of the solved Sudoku
        """
        if len(sudoku_string) != 81:
            return "❌ Invalid Sudoku format. Please provide exactly 81 characters (9x9 grid)."
            
        if not all(c.isdigit() for c in sudoku_string):
            return "❌ Invalid Sudoku format. Only digits 0-9 are allowed."
            
        try:
            # Convert input to the format expected by the model
            # The model expects: 1 for empty cells, 2-10 for digits 1-9
            input_values = []
            for c in sudoku_string:
                digit = int(c)
                if digit == 0:
                    input_values.append(1)  # Empty cell -> 1
                else:
                    input_values.append(digit + 1)  # Digit 1-9 -> 2-10
            
            # Create batch with proper format
            batch = {
                'inputs': torch.tensor(input_values, dtype=torch.long).unsqueeze(0).to(self.device),
                'labels': torch.zeros(81, dtype=torch.long).unsqueeze(0).to(self.device),  # Dummy labels
                'puzzle_identifiers': torch.tensor([0], dtype=torch.long).to(self.device)
            }
            
            with torch.inference_mode():
                # Initialize carry state
                carry = self.train_state.model.initial_carry(batch)
                
                # Run inference loop
                max_steps = 100  # Prevent infinite loops
                step = 0
                
                while step < max_steps:
                    carry, _, metrics, preds, all_finish = self.train_state.model(
                        carry=carry, 
                        batch=batch, 
                        return_keys=["logits"]
                    )
                    
                    if all_finish:
                        break
                    step += 1
                
                # Extract predictions
                if 'logits' in preds:
                    logits = preds['logits'].squeeze(0)  # Remove batch dimension
                    predictions = logits.argmax(dim=-1)
                else:
                    # Fallback: just return the processed input
                    predictions = batch['inputs'].squeeze(0)
                
            # Convert back to readable format
            solution_values = []
            for pred in predictions:
                pred_val = pred.item()
                if pred_val == 1:
                    solution_values.append('0')  # Empty cell
                elif 2 <= pred_val <= 10:
                    solution_values.append(str(pred_val - 1))  # Digits 1-9
                else:
                    solution_values.append('?')  # Unknown/invalid
            
            solution_string = "".join(solution_values)
            
            # Check if this looks like a valid solution
            if '0' in solution_string:
                return f"🤔 Partial solution (model may need more training steps):\n{self._format_sudoku(solution_string)}"
            else:
                return f"🎯 Solved Sudoku:\n{self._format_sudoku(solution_string)}"
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"❌ Error solving Sudoku: {str(e)}\n\nDetails:\n{error_details}"
    
    def _format_sudoku(self, sudoku_string: str) -> str:
        """Format a Sudoku string into a readable 9x9 grid."""
        if len(sudoku_string) != 81:
            return sudoku_string
            
        formatted = ""
        for i in range(9):
            row = sudoku_string[i*9:(i+1)*9]
            formatted += " ".join(row) + "\n"
            if i in [2, 5]:  # Add horizontal lines
                formatted += "─" * 17 + "\n"
        return formatted
    
    def solve_puzzle(self, puzzle_input: str, puzzle_type: str = "auto") -> str:
        """
        General puzzle solver that can handle different types of puzzles.
        
        Args:
            puzzle_input: String representation of the puzzle
            puzzle_type: Type of puzzle ("sudoku", "maze", "arc", or "auto" for auto-detection)
        
        Returns:
            String representation of the solution
        """
        if puzzle_type == "auto":
            puzzle_type = self._detect_puzzle_type(puzzle_input)
            
        if puzzle_type == "sudoku":
            return self.solve_sudoku(puzzle_input)
        else:
            return f"🚧 Puzzle type '{puzzle_type}' is not yet implemented in this interface."
    
    def _detect_puzzle_type(self, puzzle_input: str) -> str:
        """Auto-detect the type of puzzle based on input format."""
        if len(puzzle_input) == 81 and all(c.isdigit() for c in puzzle_input):
            return "sudoku"
        else:
            return "unknown"
    
    def run_interactive(self):
        """Run the interactive chat interface."""
        print("\n🎯 Welcome to the HRM Chat Interface!")
        print("=" * 50)
        print("📋 Supported puzzle types:")
        print("  • Sudoku: Enter 81 digits (0 for empty cells)")
        print("  • Type 'help' for more information")
        print("  • Type 'quit' or 'exit' to stop")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n🧩 Enter your puzzle (or command): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Thanks for using HRM Chat Interface!")
                    break
                    
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                    
                elif user_input.lower() == 'example':
                    self._show_examples()
                    continue
                    
                elif not user_input:
                    print("⚠️  Please enter a puzzle or command.")
                    continue
                
                # Process the puzzle
                print("🤔 Processing your puzzle...")
                result = self.solve_puzzle(user_input)
                print(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using HRM Chat Interface!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {str(e)}")
    
    def _show_help(self):
        """Display help information."""
        print("\n📖 HRM Chat Interface Help")
        print("=" * 30)
        print("🎯 Commands:")
        print("  • help     - Show this help message")
        print("  • example  - Show example puzzles")
        print("  • quit/exit- Exit the interface")
        print("\n🧩 Puzzle Formats:")
        print("  • Sudoku: 81 digits, 0 for empty cells")
        print("    Example: 003020600900305001001806400008102900700000008006708200002609500800203009005010300")
        print("\n💡 Tips:")
        print("  • The model was trained on challenging puzzles")
        print("  • Make sure your input format matches the expected format")
        print("  • Use 0 for empty cells in Sudoku puzzles")
        
    def _show_examples(self):
        """Show example puzzles."""
        print("\n📝 Example Puzzles")
        print("=" * 20)
        print("🔢 Sudoku Example:")
        print("Input:  003020600900305001001806400008102900700000008006708200002609500800203009005010300")
        print("Visual representation:")
        example = "003020600900305001001806400008102900700000008006708200002609500800203009005010300"
        print(self._format_sudoku(example))


def main():
    parser = argparse.ArgumentParser(description="HRM Chat Interface")
    parser.add_argument("--checkpoint", "-c", 
                       default="checkpoints/hrm-sudoku-extreme/checkpoint",
                       help="Path to the model checkpoint")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found at {args.checkpoint}")
        print("💡 Available checkpoints:")
        checkpoints_dir = "checkpoints"
        if os.path.exists(checkpoints_dir):
            for item in os.listdir(checkpoints_dir):
                checkpoint_path = os.path.join(checkpoints_dir, item, "checkpoint")
                if os.path.exists(checkpoint_path):
                    print(f"  • {checkpoint_path}")
        sys.exit(1)
    
    try:
        interface = HRMChatInterface(args.checkpoint)
        interface.run_interactive()
    except Exception as e:
        print(f"❌ Failed to initialize interface: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()