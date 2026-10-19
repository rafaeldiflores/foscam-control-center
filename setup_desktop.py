import os
import shutil
import subprocess

desktop = r"C:\Users\56936\OneDrive\Escritorio"
dist_exe = r"C:\app camara\dist\Foscam Control Center.exe"
app_exe = r"C:\app camara\Foscam Control Center.exe"
scratch_exe = r"C:\Users\56936\.gemini\antigravity\scratch\foscam_control_center\Foscam Control Center.exe"
desktop_exe = os.path.join(desktop, "Foscam Control Center.exe")
desktop_bat = os.path.join(desktop, "Iniciar Foscam.bat")

# 1. Copy executable to all locations
shutil.copyfile(dist_exe, app_exe)
shutil.copyfile(dist_exe, scratch_exe)
shutil.copyfile(dist_exe, desktop_exe)
print(f"Copied executable to {desktop_exe}")

# 2. Write Iniciar Foscam.bat
bat_lines = [
    "@echo off",
    "title Foscam Control Center",
    'cd /d "C:\\app camara"',
    'start "" "C:\\app camara\\Foscam Control Center.exe"',
    "exit"
]
with open(desktop_bat, "w", encoding="utf-8") as f:
    f.write("\n".join(bat_lines) + "\n")
print(f"Created batch launcher at {desktop_bat}")

# 3. Create Windows shortcut (.lnk) on Desktop via PowerShell COM
ps_script = f"""
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{desktop}\\Foscam Control Center.lnk")
$Shortcut.TargetPath = "{app_exe}"
$Shortcut.WorkingDirectory = "C:\\app camara"
$Shortcut.IconLocation = "C:\\app camara\\camera.ico"
$Shortcut.Description = "Centro de Control Foscam FI9821W V2"
$Shortcut.Save()
"""
subprocess.run(["powershell", "-Command", ps_script], check=True)
print("Created Windows Shortcut (.lnk) on Desktop with camera icon!")
