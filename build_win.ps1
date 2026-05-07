Write-Host "Building StemForge for Windows..."

# Set up Python virtual environment if it doesn't exist
if (-Not (Test-Path "build_env")) {
    Write-Host "Setting up build environment in build_env..."
    python -m venv build_env
}

# Activate virtual environment implicitly by using its pip/python executables
$pipExe = "build_env\Scripts\pip.exe"
$pythonExe = "build_env\Scripts\python.exe"

Write-Host "Installing required dependencies for StemForge..."
& $pythonExe -m pip install --upgrade pip
if (Test-Path "requirements.txt") {
    & $pipExe install -r requirements.txt
} else {
    & $pipExe install flask demucs soundfile deepfilterlib pyinstaller resampy
}

Write-Host "Patching deepfilternet to avoid ModuleNotFoundError on torchaudio >= 2.1.0 during PyInstaller analysis..."
$dfIoPath = "build_env\Lib\site-packages\df\io.py"
if (Test-Path $dfIoPath) {
    (Get-Content $dfIoPath) -replace 'from torchaudio\.backend\.common import AudioMetaData', 'class AudioMetaData: pass' | Set-Content $dfIoPath
}

Write-Host "Running PyInstaller..."
& $pythonExe -m PyInstaller --clean --noconfirm `
  --name "StemForge" `
  --onedir `
  --windowed `
  --hidden-import="torch" `
  --hidden-import="torchaudio" `
  --collect-submodules="torchaudio" `
  --hidden-import="demucs" `
  --hidden-import="demucs.__main__" `
  --collect-submodules="demucs" `
  --hidden-import="df" `
  --collect-submodules="df" `
  --hidden-import="soundfile" `
  --hidden-import="resampy" `
  --collect-data="demucs" `
  --collect-data="df" `
  stemforge_web.py

# Clean up build artifacts
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "StemForge.spec") { Remove-Item -Force "StemForge.spec" }

Write-Host "Packaging into a ZIP archive..."
$stagingDir = "win_staging"
if (Test-Path $stagingDir) { Remove-Item -Recurse -Force $stagingDir }
New-Item -ItemType Directory -Force -Path $stagingDir | Out-Null
Copy-Item -Recurse "dist\StemForge" "$stagingDir\StemForge"

if (Test-Path "StemForge-Windows.zip") { Remove-Item -Force "StemForge-Windows.zip" }
Compress-Archive -Path "$stagingDir\StemForge\*" -DestinationPath "StemForge-Windows.zip"

Remove-Item -Recurse -Force $stagingDir

Write-Host "Build complete! You can distribute StemForge-Windows.zip."
