"""
Save Server for Makeshit V2

Receives exported MJCF and STL files from the web app and saves them to output/

Usage:
    python save_server.py
"""

import os
import json
import base64
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

OUTPUT_DIR = Path(__file__).parent.parent / "output"
MESHES_DIR = OUTPUT_DIR / "meshes"
PORT = 3001


class SaveHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle preflight CORS requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        """Save exported files."""
        if self.path == "/save":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            
            try:
                data = json.loads(body)
                file_type = data.get("type")
                
                # Ensure directories exist
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                MESHES_DIR.mkdir(parents=True, exist_ok=True)
                
                if file_type == "mjcf":
                    # Save MJCF XML
                    content = data.get("content", "")
                    model_path = OUTPUT_DIR / "model.xml"
                    with open(model_path, "w") as f:
                        f.write(content)
                    print(f"✅ Saved MJCF to {model_path}")
                    
                elif file_type == "stl":
                    # Save STL mesh (base64 encoded)
                    filename = data.get("filename", "mesh.stl")
                    content_b64 = data.get("content", "")
                    content = base64.b64decode(content_b64)
                    mesh_path = MESHES_DIR / filename
                    with open(mesh_path, "wb") as f:
                        f.write(content)
                    print(f"✅ Saved mesh to {mesh_path}")
                
                else:
                    raise ValueError(f"Unknown file type: {file_type}")
                
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode())
                
            except Exception as e:
                print(f"❌ Error saving: {e}")
                self.send_response(500)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[SaveServer] {args[0]}")


def main():
    print("=" * 50)
    print("  Makeshit V2 - Save Server")
    print("=" * 50)
    print(f"\n📂 Output directory: {OUTPUT_DIR}")
    print(f"📁 Meshes directory: {MESHES_DIR}")
    print(f"🌐 Listening on http://localhost:{PORT}")
    print("\nWaiting for exports from web app (click 'Simulate' button)...")
    
    server = HTTPServer(("0.0.0.0", PORT), SaveHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹ Server stopped")


if __name__ == "__main__":
    main()
