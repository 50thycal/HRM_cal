#!/usr/bin/env python3
"""
Web-based Chat Interface for HRM using Gradio
"""

import gradio as gr
import torch
from chat_with_hrm import HRMChatInterface, ChatConfig

class HRMWebInterface:
    def __init__(self):
        self.interfaces = {}
        self.current_model = None
        
    def load_model(self, model_type):
        """Load or switch to a different model"""
        if model_type not in self.interfaces:
            print(f"Loading {model_type} model...")
            config = ChatConfig(
                model_type=model_type,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            self.interfaces[model_type] = HRMChatInterface(config)
        
        self.current_model = model_type
        return f"✅ {model_type.upper()} model loaded!"
    
    def solve_puzzle(self, model_type, puzzle_input):
        """Solve the given puzzle"""
        try:
            # Load model if needed
            if model_type not in self.interfaces:
                status = self.load_model(model_type)
                yield status + "\n\nSolving puzzle...\n"
            
            # Get the interface
            interface = self.interfaces[model_type]
            
            # Parse input
            input_tensor = interface.parse_puzzle_input(puzzle_input)
            
            # Solve
            solution = interface.solve_puzzle(input_tensor)
            
            # Format output
            formatted_solution = interface.format_output(solution)
            
            # Return both input and solution for display
            result = f"📥 Input:\n{puzzle_input}\n\n"
            result += f"✨ Solution:\n{formatted_solution}"
            
            yield result
            
        except Exception as e:
            yield f"❌ Error: {str(e)}"
    
    def get_example(self, model_type):
        """Get example for the selected model type"""
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
        return examples.get(model_type, "")
    
    def get_instructions(self, model_type):
        """Get instructions for the selected model type"""
        instructions = {
            "sudoku": """**Sudoku Instructions:**
- Enter 81 characters (9x9 grid)
- Use 0 or . for empty cells
- Use 1-9 for filled cells
- No spaces or line breaks needed""",
            
            "maze": """**Maze Instructions:**
- Use # for walls
- Use S for start position
- Use G for goal position
- Use space for empty paths
- Maximum size: 30x30""",
            
            "arc": """**ARC Instructions:**
- Enter a grid of digits (0-9)
- Each digit represents a color
- Can use spaces between numbers
- Or paste JSON format"""
        }
        return instructions.get(model_type, "")

def create_interface():
    """Create the Gradio interface"""
    web_interface = HRMWebInterface()
    
    with gr.Blocks(title="HRM Puzzle Solver", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🤖 Hierarchical Reasoning Model (HRM) Puzzle Solver
            
            Solve complex puzzles using AI! Choose a puzzle type and enter your puzzle below.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Radio(
                    ["sudoku", "maze", "arc"],
                    label="Puzzle Type",
                    value="sudoku",
                    info="Select the type of puzzle to solve"
                )
                
                instructions = gr.Markdown(web_interface.get_instructions("sudoku"))
                
                load_btn = gr.Button("Load Example", variant="secondary")
                
            with gr.Column(scale=2):
                puzzle_input = gr.Textbox(
                    label="Puzzle Input",
                    placeholder="Enter your puzzle here...",
                    lines=10,
                    max_lines=30
                )
                
                solve_btn = gr.Button("🧩 Solve Puzzle", variant="primary", size="lg")
                
                output = gr.Textbox(
                    label="Solution",
                    lines=15,
                    max_lines=30,
                    interactive=False
                )
        
        # Update instructions when model type changes
        model_selector.change(
            fn=web_interface.get_instructions,
            inputs=[model_selector],
            outputs=[instructions]
        )
        
        # Load example
        load_btn.click(
            fn=web_interface.get_example,
            inputs=[model_selector],
            outputs=[puzzle_input]
        )
        
        # Solve puzzle
        solve_btn.click(
            fn=web_interface.solve_puzzle,
            inputs=[model_selector, puzzle_input],
            outputs=[output]
        )
        
        gr.Markdown(
            """
            ---
            
            ### 📝 Notes:
            - First run will download the model checkpoint (~100MB)
            - GPU is used automatically if available
            - Processing may take a few seconds
            
            ### 🎯 Tips:
            - Click "Load Example" to see the correct format
            - For Sudoku: Enter exactly 81 characters
            - For Maze: Make sure to include both S (start) and G (goal)
            - For ARC: Use consistent spacing in your grid
            """
        )
    
    return demo

if __name__ == "__main__":
    # Check if gradio is installed
    try:
        import gradio
    except ImportError:
        print("❌ Gradio not installed. Please install it first:")
        print("   pip install gradio")
        exit(1)
    
    print("🚀 Starting HRM Web Interface...")
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create a public link
        inbrowser=True
    )