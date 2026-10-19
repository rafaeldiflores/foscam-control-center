"""
Foscam Control Center Desktop Launcher
Silently starts FastAPI server and Cloudflare tunnel if not already active,
then launches the control panel in the user's default browser.
"""

import os
import sys
import time
import subprocess
import webbrowser
import urllib.request

def is_server_running(url="http://127.0.0.1:8000/api/status"):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FoscamLauncher"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False

def is_process_running(process_name):
    try:
        CREATE_NO_WINDOW = 0x08000000
        out = subprocess.check_output(
            ["tasklist", "/FI", f"IMAGENAME eq {process_name}"],
            creationflags=CREATE_NO_WINDOW,
            text=True
        )
        return process_name.lower() in out.lower()
    except Exception:
        return False

def main():
    app_dir = r"C:\app camara"
    if not os.path.exists(app_dir):
        app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)

    CREATE_NO_WINDOW = 0x08000000

    # 1. Start app.py if not already active
    if not is_server_running():
        # Find Python executable
        python_exe = sys.executable
        if not python_exe or not os.path.exists(python_exe) or "Foscam" in python_exe:
            python_exe = "python"

        subprocess.Popen(
            [python_exe, "app.py"],
            cwd=app_dir,
            creationflags=CREATE_NO_WINDOW
        )

        # Wait up to 6 seconds for server readiness
        start = time.time()
        while time.time() - start < 6.0:
            if is_server_running():
                break
            time.sleep(0.4)

    # 2. Start Cloudflare Tunnel if not already running
    if not is_process_running("cloudflared.exe"):
        cloudflared_path = os.path.join(app_dir, "cloudflared.exe")
        if os.path.exists(cloudflared_path):
            subprocess.Popen(
                [cloudflared_path, "tunnel", "--url", "http://127.0.0.1:8000"],
                cwd=app_dir,
                creationflags=CREATE_NO_WINDOW
            )

    # 3. Open browser
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    main()
