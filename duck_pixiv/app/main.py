"""Main entrypoint for Duck Pixiv Assistant."""
import os
import sys
import json
import socket
import webbrowser
import threading
import time
import uvicorn

# Add project root and duck_pixiv directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ui.web_app import app

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if a port is currently occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def find_available_port(start_port: int = 9123, max_attempts: int = 20) -> int:
    """Finds an unused port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        if not is_port_in_use(p):
            return p
    return start_port

def open_browser(url: str):
    time.sleep(1.2)
    try:
        webbrowser.open(url)
    except Exception:
        pass

def main():
    settings_path = os.path.join(BASE_DIR, "data", "settings.json")
    host = "127.0.0.1"
    desired_port = 9123

    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
                server_conf = settings.get("server", {})
                host = server_conf.get("host", host)
                desired_port = server_conf.get("port", desired_port)
        except Exception:
            pass

    # Find free port to completely avoid any conflicts
    port = find_available_port(desired_port)

    url = f"http://{host}:{port}"
    print("=" * 60)
    print(" Duck Pixiv Assistant - Pixiv Posting Assistant")
    print(f" Web Interface: {url}")
    print(f" Port: {port} (Dedicated, Conflict-Free)")
    print(" Press Ctrl+C in this terminal to stop.")
    print("=" * 60)

    # Automatically launch browser tab in background thread
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
