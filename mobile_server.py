#!/usr/bin/env python3
"""
Simple Mobile Web Server for HRM Puzzle Solver
Works with Python's built-in libraries - no additional dependencies needed
"""

import http.server
import socketserver
import json
import os
import sys
from urllib.parse import urlparse, parse_qs

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import the HRM interface - we'll handle if it's not available
try:
    from chat_with_hrm import HRMChatInterface, ChatConfig
    import torch
    HRM_AVAILABLE = True
except ImportError:
    HRM_AVAILABLE = False
    print("⚠️  Warning: HRM modules not available. Using mock solver for demonstration.")

class PuzzleSolverHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler for puzzle solving"""
    
    def __init__(self, *args, **kwargs):
        # Initialize HRM interfaces
        self.interfaces = {}
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/' or self.path == '/mobile_interface.html':
            # Serve the mobile interface
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            with open('mobile_interface.html', 'rb') as f:
                self.wfile.write(f.read())
        else:
            # Default handler for other files
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/solve':
            # Get content length
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                # Parse JSON data
                data = json.loads(post_data.decode('utf-8'))
                puzzle_type = data.get('type')
                puzzle_input = data.get('puzzle')
                
                # Solve the puzzle
                if HRM_AVAILABLE:
                    solution = self.solve_with_hrm(puzzle_type, puzzle_input)
                else:
                    solution = self.mock_solve(puzzle_type, puzzle_input)
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = {'solution': solution}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                # Send error response
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = {'error': str(e)}
                self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_error(404)
    
    def solve_with_hrm(self, puzzle_type, puzzle_input):
        """Solve puzzle using HRM"""
        # Get or create interface
        if puzzle_type not in self.interfaces:
            print(f"Loading {puzzle_type} model...")
            config = ChatConfig(
                model_type=puzzle_type,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            self.interfaces[puzzle_type] = HRMChatInterface(config)
        
        interface = self.interfaces[puzzle_type]
        
        # Parse and solve
        input_tensor = interface.parse_puzzle_input(puzzle_input)
        solution = interface.solve_puzzle(input_tensor)
        return interface.format_output(solution)
    
    def mock_solve(self, puzzle_type, puzzle_input):
        """Mock solver for demonstration"""
        if puzzle_type == 'sudoku':
            return """5 3 4 | 6 7 8 | 9 1 2
6 7 2 | 1 9 5 | 3 4 8
1 9 8 | 3 4 2 | 5 6 7
------+-------+------
8 5 9 | 7 6 1 | 4 2 3
4 2 6 | 8 5 3 | 7 9 1
7 1 3 | 9 2 4 | 8 5 6
------+-------+------
9 6 1 | 5 3 7 | 2 8 4
2 8 7 | 4 1 9 | 6 3 5
3 4 5 | 2 8 6 | 1 7 9"""
        elif puzzle_type == 'maze':
            return """Path found! (S → G)
Steps: Right, Right, Down, Down, Down, Down, Right, Right, Right, Right, Down, Down"""
        elif puzzle_type == 'arc':
            return """Pattern detected: Symmetrical rotation
Output grid:
0 0 1 1 0
0 2 2 2 0
1 2 3 2 1
0 2 2 2 0
0 0 1 1 0"""
        else:
            return "Unknown puzzle type"
    
    def log_message(self, format, *args):
        """Override to reduce logging"""
        if '/solve' in args[0]:
            print(f"🧩 Solving {args[0].split()[1]} request")
        elif '/' == args[0].split()[1]:
            print(f"📱 Mobile interface accessed")

def get_local_ip():
    """Get the local IP address"""
    import socket
    try:
        # Create a socket to get the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def main():
    PORT = 8000
    
    # Get local IP
    local_ip = get_local_ip()
    
    # Create server
    with socketserver.TCPServer(("", PORT), PuzzleSolverHandler) as httpd:
        print("🚀 HRM Mobile Server started!")
        print(f"\n📱 To access from your iPhone:")
        print(f"   1. Make sure your iPhone is on the same WiFi network")
        print(f"   2. Open Safari on your iPhone")
        print(f"   3. Go to: http://{local_ip}:{PORT}")
        print(f"\n💻 Or access locally at: http://localhost:{PORT}")
        print(f"\n✋ Press Ctrl+C to stop the server\n")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    main()