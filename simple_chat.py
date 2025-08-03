#!/usr/bin/env python3
"""
Simple Chat Interface for HRM - Simplified Version

This script provides an easy way to interact with the trained HRM model
without complex dependencies.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import argparse
from typing import Dict, List, Optional, Any

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.functions import load_model_class


class SimpleHRMInterface:
    def __init__(self, checkpoint_path: str):
        """Initialize the interface with a pre-trained model."""
        self.checkpoint_path = checkpoint_path
        self.config = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🧠 Initializing Simple HRM Interface...")
        print(f"📱 Device: {self.device}")
        
        self._load_model()
        
    def _load_model(self):
        """Load the pre-trained model and configuration."""
        # Load configuration
        config_path = os.path.join(os.path.dirname(self.checkpoint_path), "all_config.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        print(f"📋 Loaded config: {self.config['arch']['name']}")
        
        # Initialize model architecture directly
        model_class = load_model_class(self.config['arch']['name'])
        
        # Create model config based on YAML config
        model_config = model_class.Config(**{
            'batch_size': 1,
            'seq_len': 81,
            'puzzle_emb_ndim': self.config['arch']['puzzle_emb_ndim'],
            'num_puzzle_identifiers': 1,
            'vocab_size': 11,  # For Sudoku: PAD + 0-9 encoded as 1-10
            'H_cycles': self.config['arch']['H_cycles'],
            'L_cycles': self.config['arch']['L_cycles'],
            'H_layers': self.config['arch']['H_layers'],
            'L_layers': self.config['arch']['L_layers'],
            'hidden_size': self.config['arch']['hidden_size'],
            'expansion': self.config['arch']['expansion'],
            'num_heads': self.config['arch']['num_heads'],
            'pos_encodings': self.config['arch']['pos_encodings'],
            'halt_max_steps': self.config['arch']['halt_max_steps']
        })
        
        # Create model
        self.model = model_class(model_config).to(self.device)
        
        # Load checkpoint
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            # Try to handle both regular and torch.compile checkpoints
            if any(k.startswith("_orig_mod.") for k in checkpoint.keys()):
                checkpoint = {k.removeprefix("_orig_mod."): v for k, v in checkpoint.items()}
            self.model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            print(f"⚠️  Warning: Could not load full checkpoint: {e}")
            print("    Continuing with random weights...")
        
        self.model.eval()
        print(f"✅ Model loaded successfully!")
        
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
            
            # Create input tensor
            inputs = torch.tensor(input_values, dtype=torch.long).unsqueeze(0).to(self.device)
            puzzle_identifiers = torch.tensor([0], dtype=torch.long).to(self.device)
            
            with torch.inference_mode():
                # Try simple forward pass first
                try:
                    # Method 1: Direct forward pass
                    batch = {
                        'inputs': inputs,
                        'puzzle_identifiers': puzzle_identifiers
                    }
                    
                    # Get initial carry
                    carry = self.model.initial_carry(batch)
                    
                    # Run inference loop
                    max_steps = 50
                    step = 0
                    
                    while step < max_steps:
                        try:
                            carry, _, metrics, preds, all_finish = self.model(
                                carry=carry, 
                                batch=batch, 
                                return_keys=["logits"]
                            )
                            
                            if all_finish:
                                break
                        except Exception as e:
                            print(f"⚠️  Step {step} failed: {e}")
                            break
                        step += 1
                    
                    # Extract predictions
                    if 'logits' in preds:
                        logits = preds['logits'].squeeze(0)
                        predictions = logits.argmax(dim=-1)
                    else:
                        predictions = inputs.squeeze(0)
                        
                except Exception as e:
                    print(f"⚠️  Model inference failed: {e}")
                    print("    Returning input (this suggests the model needs to be properly trained)")
                    predictions = inputs.squeeze(0)
                
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
            if solution_string == sudoku_string:
                return f"🤔 Model returned the same puzzle (may need more training or different approach):\n{self._format_sudoku(solution_string)}"
            elif '0' in solution_string:
                return f"🤔 Partial solution:\n{self._format_sudoku(solution_string)}"
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
    
    def run_interactive(self):
        """Run the interactive chat interface."""
        print("\n🎯 Welcome to the Simple HRM Interface!")
        print("=" * 50)
        print("📋 Supported puzzle types:")
        print("  • Sudoku: Enter 81 digits (0 for empty cells)")
        print("  • Type 'help' for more information")
        print("  • Type 'example' to see an example")
        print("  • Type 'quit' or 'exit' to stop")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n🧩 Enter your Sudoku puzzle: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Thanks for using Simple HRM Interface!")
                    break
                    
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                    
                elif user_input.lower() == 'example':
                    self._show_example()
                    continue
                    
                elif not user_input:
                    print("⚠️  Please enter a puzzle or command.")
                    continue
                
                # Process the puzzle
                print("🤔 Processing your puzzle...")
                result = self.solve_sudoku(user_input)
                print(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using Simple HRM Interface!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {str(e)}")
    
    def _show_help(self):
        """Display help information."""
        print("\n📖 Simple HRM Interface Help")
        print("=" * 30)
        print("🎯 Commands:")
        print("  • help     - Show this help message")
        print("  • example  - Show an example Sudoku")
        print("  • quit/exit- Exit the interface")
        print("\n🧩 Sudoku Format:")
        print("  • Enter exactly 81 digits")
        print("  • Use 0 for empty cells")
        print("  • Use digits 1-9 for filled cells")
        print("  • Example: 003020600900305001001806400008102900700000008006708200002609500800203009005010300")
        
    def _show_example(self):
        """Show an example Sudoku."""
        print("\n📝 Example Sudoku Puzzle")
        print("=" * 25)
        example = "003020600900305001001806400008102900700000008006708200002609500800203009005010300"
        print(f"Input: {example}")
        print("Visual representation:")
        print(self._format_sudoku(example))
        print("💡 You can copy and paste this example to test the interface!")


def main():
    parser = argparse.ArgumentParser(description="Simple HRM Chat Interface")
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
        interface = SimpleHRMInterface(args.checkpoint)
        interface.run_interactive()
    except Exception as e:
        print(f"❌ Failed to initialize interface: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()