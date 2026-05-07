# StemForge 🎛️

StemForge is a powerful, standalone macOS application that uses state-of-the-art AI to separate audio tracks into high-quality stems (vocals, drums, bass, and other). It combines **Demucs** for core source separation and **DeepFilterNet** for post-processing and vocal enhancement.

## Features
- **Local AI Processing:** Runs entirely on your Mac using PyTorch (optimized for Apple Silicon / MPS).
- **Multiple Demucs Models:** Choose between HTDemucs (4-stem), HTDemucs 6-stem (adds guitar & piano), HTDemucs Fine-Tuned for Vocals, and MDX-Net.
- **DeepFilterNet Integration:** Automatically enhances and cleans separated stems using DeepFilterNet.
- **Web UI:** A clean, local web interface to manage your audio files, select models, and monitor processing.

## Installation

You do not need to install Python or set up any environments to use StemForge!

1. Go to the [Releases](../../releases) page.
2. Download the latest `StemForge.dmg` file.
3. Open the `.dmg` and drag the `StemForge.app` into your `/Applications` folder.
4. **Important:** Because this is an open-source app without a paid Apple developer signature, macOS Gatekeeper may block it on first run. To bypass this, **Right-click** (or Control-click) `StemForge.app` in your Applications folder and select **Open**.

## Development & Building from Source

If you want to modify StemForge or build it yourself:

### Prerequisites
- macOS (Apple Silicon recommended)
- Homebrew
- Python 3.11 (`brew install python@3.11`)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/JohnnyRaha/StemForge.git
   cd StemForge
   ```

2. Run the development server (it will self-bootstrap a virtual environment):
   ```bash
   python3.11 stemforge_web.py
   ```

### Building the macOS App
To compile StemForge into a standalone `.app` and package it into a `.dmg`:

```bash
bash build_mac.sh
```

This script will:
- Set up a clean `build_env`.
- Install all dependencies from `requirements.txt`.
- Apply necessary patches (like Torchaudio backend mocks for DeepFilterNet compatibility).
- Use **PyInstaller** to compile the app.
- Package everything into a `.dmg` file ready for distribution.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
