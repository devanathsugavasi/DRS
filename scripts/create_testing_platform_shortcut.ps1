$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $Root "scripts\start_testing_platform.ps1"
$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Cricket DRS Testing Platform.lnk"

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$Launcher`""
$Shortcut.WorkingDirectory = $Root
$Shortcut.IconLocation = "powershell.exe,0"
$Shortcut.Description = "Start the offline Cricket DRS Testing Platform"
$Shortcut.Save()

Write-Host "Shortcut created: $ShortcutPath"
Write-Host "To pin it to the taskbar, right-click the shortcut and choose Pin to taskbar."
