Write-Host "--- StemForge Windows Build System ---"

# Detect if we are in CI
$isCI = $false
if ($env:GITHUB_ACTIONS -eq "true") {
    Write-Host "INFO: Running in GitHub Actions CI environment."
    $isCI = $true
}

# Environment Setup
if ($isCI) {
    Write-Host "INFO: Skipping venv creation in CI (using pre-installed environment)."
    $pipExe = "pip"
    $pythonExe = "python"
} else {
    # Set up Python virtual environment if it doesn't exist
    if (-Not (Test-Path "build_env")) {
        Write-Host "Setting up build environment in build_env..."
        python -m venv build_env
    }
    $pipExe = "build_env\Scripts\pip.exe"
    $pythonExe = "build_env\Scripts\python.exe"
}

Write-Host "Installing/Updating required dependencies..."
& $pythonExe -m pip install --upgrade pip
if (Test-Path "requirements.txt") {
    & $pipExe install -r requirements.txt
} else {
    & $pipExe install flask demucs soundfile deepfilterlib deepfilternet==0.5.6 pyinstaller resampy
}

Write-Host "Applying DeepFilterNet patches..."
# Dynamically locate df/io.py
$dfIoPath = & $pythonExe -c "import df.io as dio; print(dio.__file__)" 2>$null
if (Test-Path $dfIoPath) {
    Write-Host "Patching $dfIoPath..."
    (Get-Content $dfIoPath) -replace 'from torchaudio\.backend\.common import AudioMetaData', 'class AudioMetaData: pass' | Set-Content $dfIoPath
} else {
    Write-Host "WARNING: Could not locate df/io.py for patching. DeepFilterNet might fail during PyInstaller analysis."
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
  --collect-all="antlr4" `
  --collect-all="df" `
  --collect-all="deepfilterlib" `
  --hidden-import="soundfile" `
  --hidden-import="resampy" `
  --collect-data="demucs" `
  stemforge_web.py

# Verify build success
if (-Not (Test-Path "dist\StemForge")) {
    Write-Host "ERROR: PyInstaller failed to create StemForge folder"
    exit 1
}

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

Write-Host "Build complete! Final artifact: StemForge-Windows.zip"

