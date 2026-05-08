#!/bin/bash
set -e

# Change to the directory of the script
cd "$(dirname "$0")"

echo "--- StemForge macOS Build System ---"

# Detect if we are in CI
IF_CI=false
if [ "$GITHUB_ACTIONS" = "true" ]; then
  echo "INFO: Running in GitHub Actions CI environment."
  IF_CI=true
fi

# Find a suitable Python 3.11
if [ "$IF_CI" = "true" ]; then
  # In CI, we assume setup-python has provided the right version
  PY311=$(command -v python3 || command -v python)
else
  if command -v python3.11 >/dev/null 2>&1; then
    PY311=$(command -v python3.11)
  elif [ -x "/opt/homebrew/bin/python3.11" ]; then
    PY311="/opt/homebrew/bin/python3.11"
  elif [ -x "/opt/homebrew/opt/python@3.11/bin/python3.11" ]; then
    PY311="/opt/homebrew/opt/python@3.11/bin/python3.11"
  elif [ -x "/usr/local/bin/python3.11" ]; then
    PY311="/usr/local/bin/python3.11"
  else
    echo "ERROR: Python 3.11 not found. Please install Python 3.11."
    exit 1
  fi
fi

echo "Using Python: $PY311"

# Environment Setup
if [ "$IF_CI" = "true" ]; then
  echo "INFO: Skipping venv creation in CI (using pre-installed environment)."
else
  # Remove old build_env if it was created with a different Python version
  if [ -d "build_env" ]; then
    CURRENT_PY=$(build_env/bin/python3 --version 2>/dev/null || echo "unknown")
    if [[ "$CURRENT_PY" != *"3.11"* ]]; then
      echo "Removing old build_env (was $CURRENT_PY, need 3.11)..."
      rm -rf build_env
    fi
  fi

  echo "Setting up build environment in build_env (Python 3.11)..."
  "$PY311" -m venv build_env
  source build_env/bin/activate
fi

echo "Installing/Updating required dependencies..."
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  pip install flask demucs soundfile deepfilternet pyinstaller resampy
fi

echo "Applying DeepFilterNet patches..."
# Dynamically locate df/io.py to avoid hardcoded venv paths
DF_IO_PATH=$(python3 -c "import df.io as dio; print(dio.__file__)" 2>/dev/null || echo "")

if [ -f "$DF_IO_PATH" ]; then
  echo "Patching $DF_IO_PATH..."
  # Patch deepfilternet to avoid ModuleNotFoundError on torchaudio >= 2.1.0 during PyInstaller analysis
  sed -i '' 's/from torchaudio.backend.common import AudioMetaData/class AudioMetaData: pass/g' "$DF_IO_PATH"
else
  echo "WARNING: Could not locate df/io.py for patching. DeepFilterNet might fail during PyInstaller analysis."
fi

echo "Running PyInstaller..."
# We use --onedir and --windowed to create a standard macOS .app bundle
pyinstaller --clean --noconfirm \
  --name "StemForge" \
  --windowed \
  --onedir \
  --hidden-import="torch" \
  --hidden-import="torchaudio" \
  --collect-submodules="torchaudio" \
  --hidden-import="demucs" \
  --hidden-import="demucs.__main__" \
  --collect-submodules="demucs" \
  --collect-all="antlr4" \
  --collect-all="df" \
  --collect-all="deepfilterlib" \
  --hidden-import="soundfile" \
  --hidden-import="resampy" \
  --collect-data="demucs" \
  stemforge_web.py

# Verify build success
if [ ! -d "dist/StemForge.app" ]; then
  echo "ERROR: PyInstaller failed to create StemForge.app"
  exit 1
fi

# Remove PyInstaller build directories to keep workspace clean
rm -rf build
rm -f StemForge.spec

echo "Build complete! StemForge.app created in dist/ folder."

echo "Packaging into a Disk Image (DMG)..."
mkdir -p dmg_staging
cp -R dist/StemForge.app dmg_staging/
ln -s /Applications dmg_staging/Applications

# Remove old DMG if it exists
rm -f "StemForge.dmg"

hdiutil create -volname "StemForge" -srcfolder dmg_staging -ov -format UDZO StemForge.dmg
rm -rf dmg_staging

echo "Packaging complete! Final artifact: StemForge.dmg"

