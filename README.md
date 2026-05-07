# StemForge 🎛️

StemForge is a powerful, standalone macOS and Windows 11 application that uses state-of-the-art AI to separate audio tracks into high-quality stems (vocals, drums, bass, and other). It combines **Demucs** for core source separation and **DeepFilterNet** for post-processing and vocal enhancement.

## Features
- **Local AI Processing:** Runs entirely on your Mac or PC using PyTorch (optimized for Apple Silicon / MPS and NVIDIA CUDA GPUs).
- **Multiple Demucs Models:** Choose between HTDemucs (4-stem), HTDemucs 6-stem (adds guitar & piano), HTDemucs Fine-Tuned for Vocals, and MDX-Net.
- **DeepFilterNet Integration:** Automatically enhances and cleans separated stems using DeepFilterNet.
- **Web UI:** A clean, local web interface to manage your audio files, select models, and monitor processing.

## Installation

You do not need to install Python or set up any environments to use StemForge!

### macOS
1. Go to the [Releases](../../releases) page.
2. Download the latest `StemForge.dmg` file.
3. Open the `.dmg` and drag the `StemForge.app` into your `/Applications` folder.
4. **Important:** Because this is an open-source app without a paid Apple developer signature, macOS Gatekeeper may block it on first run. To bypass this, **Right-click** (or Control-click) `StemForge.app` in your Applications folder and select **Open**.

### Windows 11
1. Go to the [Releases](../../releases) page.
2. Download the latest `StemForge-Windows.zip` file.
3. Extract the ZIP file to a folder of your choice (e.g., Desktop or Documents).
4. Run `StemForge.exe` from inside the extracted folder.
   *(Note: Microsoft Defender SmartScreen might warn you the first time since the app is unsigned. Click "More info" and then "Run anyway".)*

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

### Building the App (Executables)
To compile StemForge into a standalone application, you can use the provided build scripts.

#### macOS
```bash
bash build_mac.sh
```
This will compile the app and package it into a `.dmg` file ready for distribution.

#### Windows
```powershell
.\build_win.ps1
```
This will compile the app and package it into a `StemForge-Windows.zip` archive.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
