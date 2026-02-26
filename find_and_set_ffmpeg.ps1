# Find ffmpeg installed by winget and add to User PATH (run in PowerShell)
# Run: .\find_and_set_ffmpeg.ps1

$winGetPath = "$env:LOCALAPPDATA\Microsoft\WinGet"
$found = $null

if (Test-Path $winGetPath) {
    $found = Get-ChildItem -Path $winGetPath -Recurse -Filter "ffmpeg.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
}
if (-not $found) {
    $found = Get-ChildItem -Path "C:\ffmpeg", "$env:USERPROFILE\ffmpeg", "D:\ffmpeg" -Recurse -Filter "ffmpeg.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
}

if ($found) {
    $binDir = $found.DirectoryName
    Write-Host "Found ffmpeg at: $binDir"
    $currentUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($currentUserPath -notlike "*$binDir*") {
        [Environment]::SetEnvironmentVariable("Path", "$currentUserPath;$binDir", "User")
        Write-Host "Added to User PATH. Restart this terminal (or Cursor) and run: ffmpeg -version"
    } else {
        Write-Host "Already on User PATH. If still not found, restart Cursor and open a new terminal."
    }
    # Use in THIS session immediately
    $env:Path = "$env:Path;$binDir"
    Write-Host "Current session updated. Testing:"
    & "$binDir\ffmpeg.exe" -version 2>&1 | Select-Object -First 1
} else {
    Write-Host "ffmpeg.exe not found in common locations."
    Write-Host "Check manually: $env:LOCALAPPDATA\Microsoft\WinGet\Packages"
    Write-Host "Or download from https://www.gyan.dev/ffmpeg/builds/ and unzip to e.g. C:\ffmpeg, then add C:\ffmpeg\bin to User PATH."
}
