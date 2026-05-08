#!/bin/bash
set -e

# Change to the directory of the script
cd "$(dirname "$0")"

# DeepFilterNet requires Python 3.11
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

echo "Installing required dependencies for StemForge..."
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  pip install flask demucs soundfile deepfilternet pyinstaller resampy
fi

echo "Running PyInstaller..."
# Patch deepfilternet to avoid ModuleNotFoundError on torchaudio >= 2.1.0 during PyInstaller analysis
DF_IO_PATH="build_env/lib/python3.11/site-packages/df/io.py"
if [ -f "$DF_IO_PATH" ]; then
  sed -i '' 's/from torchaudio.backend.common import AudioMetaData/class AudioMetaData: pass/g' "$DF_IO_PATH"
fi
# We use --onedir and --windowed to create a standard macOS .app bundle
# PyTorch and other deep learning libs require collecting all hidden imports and data files
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
  --collect-all="df" \
  --hidden-import="soundfile" \
  --hidden-import="resampy" \
  --collect-data="demucs" \
  stemforge_web.py

# Remove PyInstaller build directories to keep workspace clean
rm -rf build
rm StemForge.spec

echo "Build complete! You can find StemForge.app in the dist/ folder."

echo "Packaging into a Disk Image (DMG)..."
mkdir -p dmg_staging
cp -R dist/StemForge.app dmg_staging/
ln -s /Applications dmg_staging/Applications

# Remove old DMG if it exists
if [ -f "StemForge.dmg" ]; then
    rm "StemForge.dmg"
fi

hdiutil create -volname "StemForge" -srcfolder dmg_staging -ov -format UDZO StemForge.dmg
rm -rf dmg_staging

echo "Packaging complete! You can distribute StemForge.dmg."
