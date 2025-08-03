#!/usr/bin/env python3
"""
Quick test script for HRM Chat Interface
"""

import torch
from chat_with_hrm import HRMChatInterface, ChatConfig

def test_sudoku():
    """Test Sudoku solving"""
    print("Testing Sudoku solver...")
    
    config = ChatConfig(
        model_type="sudoku",
        device="cpu"  # Use CPU for testing
    )
    
    # Create interface
    chat = HRMChatInterface(config)
    
    # Test puzzle
    puzzle = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
    
    print(f"\nInput puzzle: {puzzle}")
    
    # Parse puzzle
    input_tensor = chat.parse_puzzle_input(puzzle)
    print(f"Parsed shape: {input_tensor.shape}")
    
    # Solve
    print("\nSolving...")
    solution = chat.solve_puzzle(input_tensor)
    
    # Display
    print("\nSolution:")
    print(chat.format_output(solution))
    
    print("\n✅ Sudoku test completed!")

def test_parsing():
    """Test puzzle parsing"""
    print("Testing puzzle parsing...")
    
    config = ChatConfig(model_type="sudoku", device="cpu")
    chat = HRMChatInterface(config)
    
    # Test Sudoku parsing
    sudoku_inputs = [
        "0"*81,  # All zeros
        "."*81,  # All dots
        "1234567890"*8 + "1",  # Mixed numbers
    ]
    
    for inp in sudoku_inputs:
        try:
            tensor = chat.parse_puzzle_input(inp)
            print(f"✓ Parsed Sudoku input of length {len(inp)}")
        except Exception as e:
            print(f"✗ Failed to parse: {e}")
    
    print("\n✅ Parsing test completed!")

if __name__ == "__main__":
    print("🧪 Running HRM Chat Interface Tests\n")
    
    # Check if models are available
    try:
        test_parsing()
        print("\n" + "="*60 + "\n")
        
        # Only run full test if user confirms (as it downloads models)
        response = input("Run full Sudoku solving test? This will download the model (~100MB). [y/N]: ")
        if response.lower() == 'y':
            test_sudoku()
        else:
            print("Skipping full test.")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()