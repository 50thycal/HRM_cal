#!/usr/bin/env python3
"""
Chat Interface for Hierarchical Reasoning Model (HRM)
Provides an easy-to-use conversational interface for solving puzzles with HRM.
"""

import os
import sys
import yaml
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import argparse
from huggingface_hub import hf_hub_download
import json

# Import HRM components
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class
from pretrain import PretrainConfig, init_train_state


@dataclass
class ChatConfig:
    """Configuration for the chat interface"""
    model_type: str = "sudoku"  # sudoku, maze, or arc
    checkpoint_path: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    temperature: float = 1.0
    top_k: int = 10


class HRMChatInterface:
    """Interactive chat interface for HRM model"""
    
    # Hugging Face checkpoint URLs
    CHECKPOINTS = {
        "sudoku": "sapientinc/HRM-checkpoint-sudoku-extreme",
        "maze": "sapientinc/HRM-checkpoint-maze-30x30-hard",
        "arc": "sapientinc/HRM-checkpoint-ARC-2"
    }
    
    def __init__(self, config: ChatConfig):
        self.config = config
        self.model = None
        self.train_config = None
        self.metadata = None
        
        print(f"🚀 Initializing HRM Chat Interface...")
        print(f"📋 Model type: {config.model_type}")
        print(f"🖥️  Device: {config.device}")
        
        self._load_model()
    
    def _download_checkpoint(self) -> str:
        """Download checkpoint from Hugging Face if needed"""
        if self.config.checkpoint_path and os.path.exists(self.config.checkpoint_path):
            return self.config.checkpoint_path
        
        print(f"📥 Downloading checkpoint for {self.config.model_type}...")
        repo_id = self.CHECKPOINTS.get(self.config.model_type)
        
        if not repo_id:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)
        
        # Download model files
        checkpoint_dir = f"checkpoints/{self.config.model_type}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Download the model checkpoint and config
        files_to_download = ["model.pt", "all_config.yaml"]
        
        for filename in files_to_download:
            local_path = os.path.join(checkpoint_dir, filename)
            if not os.path.exists(local_path):
                print(f"  Downloading {filename}...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=checkpoint_dir,
                    local_dir_use_symlinks=False
                )
        
        return os.path.join(checkpoint_dir, "model.pt")
    
    def _load_model(self):
        """Load the HRM model and configuration"""
        checkpoint_path = self._download_checkpoint()
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        # Load configuration
        config_path = os.path.join(checkpoint_dir, "all_config.yaml")
        with open(config_path, "r") as f:
            self.train_config = PretrainConfig(**yaml.safe_load(f))
        
        # Create dummy metadata for model initialization
        self.metadata = PuzzleDatasetMetadata(
            num_puzzles=1000,
            vocab_size=self._get_vocab_size(),
            max_seq_len=self._get_max_seq_len()
        )
        
        # Initialize model
        print("🔧 Loading model...")
        
        # Set device for initialization
        if self.config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(0)
            
        train_state = init_train_state(self.train_config, self.metadata, world_size=1)
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        if isinstance(checkpoint_data, dict) and '_orig_mod' in str(checkpoint_data.keys()):
            # Handle compiled model state dict
            checkpoint_data = {k.removeprefix("_orig_mod."): v for k, v in checkpoint_data.items()}
        
        try:
            train_state.model.load_state_dict(checkpoint_data, strict=True)
        except RuntimeError:
            # Try with strict=False if exact match fails
            train_state.model.load_state_dict(checkpoint_data, strict=False)
            print("⚠️  Loaded model with some mismatched keys (this is usually fine)")
            
        train_state.model.eval()
        train_state.model.to(self.config.device)
        
        self.model = train_state.model
        print("✅ Model loaded successfully!")
    
    def _get_vocab_size(self) -> int:
        """Get vocabulary size based on model type"""
        vocab_sizes = {
            "sudoku": 11,  # 0-9 + empty
            "maze": 5,     # wall, empty, start, goal, path
            "arc": 11      # 0-9 + background
        }
        return vocab_sizes.get(self.config.model_type, 11)
    
    def _get_max_seq_len(self) -> int:
        """Get maximum sequence length based on model type"""
        seq_lens = {
            "sudoku": 81,     # 9x9 grid
            "maze": 900,      # 30x30 grid
            "arc": 1024       # variable, max 32x32
        }
        return seq_lens.get(self.config.model_type, 1024)
    
    def parse_puzzle_input(self, input_text: str) -> torch.Tensor:
        """Parse user input into tensor format"""
        if self.config.model_type == "sudoku":
            return self._parse_sudoku(input_text)
        elif self.config.model_type == "maze":
            return self._parse_maze(input_text)
        elif self.config.model_type == "arc":
            return self._parse_arc(input_text)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def _parse_sudoku(self, input_text: str) -> torch.Tensor:
        """Parse Sudoku puzzle from text"""
        # Remove all non-digit characters and dots
        cleaned = ''.join(c if c.isdigit() or c == '.' else '' for c in input_text)
        
        if len(cleaned) != 81:
            raise ValueError(f"Sudoku puzzle must have exactly 81 cells, got {len(cleaned)}")
        
        # Convert to tensor (0 for empty cells)
        puzzle = []
        for c in cleaned:
            if c == '.' or c == '0':
                puzzle.append(0)
            else:
                puzzle.append(int(c))
        
        return torch.tensor(puzzle, dtype=torch.long).reshape(9, 9)
    
    def _parse_maze(self, input_text: str) -> torch.Tensor:
        """Parse maze from text"""
        lines = input_text.strip().split('\n')
        
        # Map characters to values
        char_map = {
            '#': 0,  # wall
            ' ': 1,  # empty
            '.': 1,  # empty (alternative)
            'S': 2,  # start
            'G': 3,  # goal
            'E': 3,  # end (alternative)
        }
        
        maze = []
        for line in lines:
            row = []
            for char in line:
                if char in char_map:
                    row.append(char_map[char])
                else:
                    row.append(1)  # default to empty
            maze.append(row)
        
        # Pad or truncate to 30x30
        maze_tensor = torch.zeros(30, 30, dtype=torch.long)
        for i in range(min(30, len(maze))):
            for j in range(min(30, len(maze[i]))):
                maze_tensor[i, j] = maze[i][j]
        
        return maze_tensor
    
    def _parse_arc(self, input_text: str) -> torch.Tensor:
        """Parse ARC puzzle from text or JSON"""
        try:
            # Try parsing as JSON first
            data = json.loads(input_text)
            if isinstance(data, list):
                grid = data
            elif isinstance(data, dict) and 'input' in data:
                grid = data['input']
            else:
                raise ValueError("Invalid ARC format")
        except json.JSONDecodeError:
            # Parse as text grid
            lines = input_text.strip().split('\n')
            grid = []
            for line in lines:
                row = [int(c) for c in line if c.isdigit()]
                grid.append(row)
        
        return torch.tensor(grid, dtype=torch.long)
    
    def format_output(self, output_tensor: torch.Tensor) -> str:
        """Format model output for display"""
        if self.config.model_type == "sudoku":
            return self._format_sudoku(output_tensor)
        elif self.config.model_type == "maze":
            return self._format_maze(output_tensor)
        elif self.config.model_type == "arc":
            return self._format_arc(output_tensor)
        else:
            return str(output_tensor)
    
    def _format_sudoku(self, tensor: torch.Tensor) -> str:
        """Format Sudoku solution"""
        grid = tensor.reshape(9, 9).cpu().numpy()
        
        result = "╔═══════╦═══════╦═══════╗\n"
        for i in range(9):
            if i % 3 == 0 and i > 0:
                result += "╠═══════╬═══════╬═══════╣\n"
            
            result += "║"
            for j in range(9):
                if j % 3 == 0 and j > 0:
                    result += "║"
                result += f" {grid[i, j] if grid[i, j] > 0 else '.'} "
            result += "║\n"
        result += "╚═══════╩═══════╩═══════╝"
        
        return result
    
    def _format_maze(self, tensor: torch.Tensor) -> str:
        """Format maze solution"""
        maze = tensor.cpu().numpy()
        
        # Character mapping
        char_map = {
            0: '█',  # wall
            1: ' ',  # empty
            2: 'S',  # start
            3: 'G',  # goal
            4: '·',  # path
        }
        
        result = ""
        for row in maze:
            for cell in row:
                result += char_map.get(int(cell), '?')
            result += "\n"
        
        return result.strip()
    
    def _format_arc(self, tensor: torch.Tensor) -> str:
        """Format ARC output"""
        grid = tensor.cpu().numpy()
        
        # Color codes for terminal
        colors = [
            '\033[40m  \033[0m',  # 0: Black
            '\033[44m  \033[0m',  # 1: Blue
            '\033[41m  \033[0m',  # 2: Red
            '\033[42m  \033[0m',  # 3: Green
            '\033[43m  \033[0m',  # 4: Yellow
            '\033[45m  \033[0m',  # 5: Magenta
            '\033[46m  \033[0m',  # 6: Cyan
            '\033[47m  \033[0m',  # 7: White
            '\033[100m  \033[0m', # 8: Gray
            '\033[101m  \033[0m', # 9: Light Red
        ]
        
        result = ""
        for row in grid:
            for cell in row:
                if 0 <= cell < len(colors):
                    result += colors[cell]
                else:
                    result += f"{cell:2}"
            result += "\n"
        
        return result
    
    def solve_puzzle(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Solve the given puzzle using HRM"""
        with torch.no_grad():
            # Prepare input
            input_tensor = input_tensor.to(self.config.device)
            
            # Create batch
            inputs = input_tensor.flatten().unsqueeze(0)  # [1, seq_len]
            
            # Create dummy labels (same shape as inputs)
            labels = inputs.clone()
            
            # Create puzzle identifier
            puzzle_identifiers = torch.zeros(1, dtype=torch.long, device=self.config.device)
            
            # Forward pass
            output = self.model(
                inputs=inputs,
                labels=labels,
                puzzle_identifiers=puzzle_identifiers
            )
            
            # Extract predictions
            logits = output['logits'][0]  # Remove batch dimension
            predictions = torch.argmax(logits, dim=-1)
            
            # Reshape to original grid shape
            if self.config.model_type == "sudoku":
                return predictions.reshape(9, 9)
            elif self.config.model_type == "maze":
                return predictions.reshape(30, 30)
            else:
                # For ARC, maintain the input shape
                original_shape = input_tensor.shape
                return predictions[:original_shape.numel()].reshape(original_shape)
    
    def chat_loop(self):
        """Main chat interaction loop"""
        print("\n" + "="*60)
        print(f"🤖 HRM {self.config.model_type.upper()} Solver")
        print("="*60)
        print("\n📝 Instructions:")
        
        if self.config.model_type == "sudoku":
            print("  - Enter a 9x9 Sudoku puzzle (81 characters)")
            print("  - Use 0 or . for empty cells")
            print("  - Example: 53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79")
        elif self.config.model_type == "maze":
            print("  - Enter a maze using:")
            print("    # = wall, S = start, G = goal, space = empty")
            print("  - Maximum size: 30x30")
        elif self.config.model_type == "arc":
            print("  - Enter an ARC puzzle grid")
            print("  - Use digits 0-9 for colors")
            print("  - Can paste JSON format or space-separated grid")
        
        print("\n💡 Commands:")
        print("  - 'help' - Show instructions")
        print("  - 'example' - Show an example puzzle")
        print("  - 'quit' or 'exit' - Exit the program")
        print("\n" + "="*60 + "\n")
        
        while True:
            try:
                user_input = input("🧩 Enter puzzle (or command): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if user_input.lower() == 'example':
                    self.show_example()
                    continue
                
                if not user_input:
                    continue
                
                # Parse and solve puzzle
                print("\n🔍 Parsing puzzle...")
                input_tensor = self.parse_puzzle_input(user_input)
                
                print("🧠 Thinking...")
                solution = self.solve_puzzle(input_tensor)
                
                print("\n✨ Solution:")
                print(self.format_output(solution))
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'help' for instructions.\n")
    
    def show_help(self):
        """Show detailed help information"""
        print("\n" + "="*60)
        print("📚 HELP")
        print("="*60)
        
        if self.config.model_type == "sudoku":
            print("\nSUDOKU FORMAT:")
            print("Enter 81 characters representing the 9x9 grid from left to right, top to bottom.")
            print("Use 0 or . for empty cells, 1-9 for filled cells.\n")
            print("Example input:")
            print("53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79")
            
        elif self.config.model_type == "maze":
            print("\nMAZE FORMAT:")
            print("Draw your maze using these characters:")
            print("  # = wall")
            print("  S = start position")
            print("  G = goal position")
            print("  space = empty path\n")
            print("Example:")
            print("#########")
            print("#S      #")
            print("# ##### #")
            print("#     # #")
            print("##### # #")
            print("#     # #")
            print("# ##### #")
            print("#      G#")
            print("#########")
            
        elif self.config.model_type == "arc":
            print("\nARC FORMAT:")
            print("Enter a grid using digits 0-9 where each digit represents a color.")
            print("You can use either format:")
            print("\n1. Space-separated grid:")
            print("0 0 1 1 0")
            print("0 2 2 2 0")
            print("1 2 3 2 1")
            print("\n2. JSON format:")
            print('{"input": [[0,0,1,1,0],[0,2,2,2,0],[1,2,3,2,1]]}')
        
        print("\n" + "="*60)
    
    def show_example(self):
        """Show an example puzzle"""
        examples = {
            "sudoku": "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79",
            "maze": """#########
#S      #
# ##### #
#     # #
##### # #
#     # #
# ##### #
#      G#
#########""",
            "arc": """0 0 1 1 0
0 2 2 2 0
1 2 3 2 1
0 2 2 2 0
0 0 1 1 0"""
        }
        
        example = examples.get(self.config.model_type, "No example available")
        print(f"\n📋 Example {self.config.model_type} puzzle:")
        print("-" * 40)
        print(example)
        print("-" * 40)
        print("Copy and paste this to try it!\n")


def main():
    parser = argparse.ArgumentParser(description="Chat with Hierarchical Reasoning Model")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["sudoku", "maze", "arc"],
        default="sudoku",
        help="Type of puzzle to solve (default: sudoku)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (will download if not provided)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)"
    )
    
    args = parser.parse_args()
    
    config = ChatConfig(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    chat = HRMChatInterface(config)
    chat.chat_loop()


if __name__ == "__main__":
    main()