#!/usr/bin/env python3
"""
Simple HTTP server for the 90s RODLA Frontend
Run this in the frontend directory to serve the frontend
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

PORT = 8080

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

def main():
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("=" * 60)
    print("🚀 RODLA 90s FRONTEND SERVER")
    print("=" * 60)
    print(f"📁 Serving from: {script_dir}")
    print(f"🌐 Server URL: http://localhost:{PORT}")
    print(f"🔗 Open in browser: http://localhost:{PORT}")
    print("\n⚠️  Backend must be running on http://localhost:8000")
    print("=" * 60)
    print("\nPress Ctrl+C to stop server\n")
    
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("🛑 SERVER STOPPED")
        print("=" * 60)
        sys.exit(0)

if __name__ == "__main__":
    main()
