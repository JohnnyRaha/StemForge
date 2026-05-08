# StemForge Web — standalone self-bootstrapping build
"""
StemForge Web — AI Audio Stem Separator
Apple Silicon · M4 Max Optimised · No terminal required

Just run:  python3 stemforge_web.py
On first launch it silently installs all dependencies into ~/.stemforge,
then opens http://localhost:7777 in your browser automatically.
"""

# ── Self-bootstrap — runs BEFORE any other import ────────────────────────────
import sys, os, subprocess
from pathlib import Path

# Support PyInstaller internal execution for subprocesses
if getattr(sys, 'frozen', False) and len(sys.argv) >= 3 and sys.argv[1] == '--internal-exec':
    script_path = sys.argv[2]
    sys.argv = [script_path] + sys.argv[3:]
    with open(script_path, 'r', encoding='utf-8') as f:
        code = f.read()
    exec(code, {'__name__': '__main__'})
    sys.exit(0)

_VENV     = Path.home() / ".stemforge"
_VENV_BIN = _VENV / "Scripts" if os.name == "nt" else _VENV / "bin"
_VENV_PY  = _VENV_BIN / ("python.exe" if os.name == "nt" else "python3")
_VENV_PIP = _VENV_BIN / ("pip.exe" if os.name == "nt" else "pip")
_MARKER   = _VENV / ".setup_complete"
_PACKAGES = ["flask", "demucs", "soundfile", "deepfilterlib", "resampy", "numpy<2"]

def _bootstrap():
    """
    Ensure we are running inside ~/.stemforge with all deps installed.
    If not, create the venv, install packages, then re-exec this script
    inside the managed venv — transparently and without user action.
    """
    # PyInstaller: skip bootstrap if frozen
    if getattr(sys, 'frozen', False):
        return

    # Already inside our managed venv → nothing to do
    if Path(sys.prefix).resolve() == _VENV.resolve():
        return

    # ── Create venv if it does not exist yet ─────────────────────────────────
    if not _VENV_PY.exists():
        _notify("StemForge · First-run setup — creating environment…")
        
        # Find Python 3.11
        import shutil
        py_exe = sys.executable
        _py_candidates = [sys.executable, "python", "python3"] if os.name == "nt" else [
            "python3.11", "/opt/homebrew/bin/python3.11", 
            "/opt/homebrew/opt/python@3.11/bin/python3.11", "/usr/local/bin/python3.11",
            "python3", sys.executable
        ]
        for p in _py_candidates:
            if shutil.which(p):
                py_exe = shutil.which(p)
                break
                
        subprocess.check_call(
            [py_exe, "-m", "venv", str(_VENV)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # ── Install packages once (marker file guards re-installs) ───────────────
    if not _MARKER.exists():
        _notify("StemForge · Installing packages — this takes ~2 min once…")
        subprocess.check_call(
            [str(_VENV_PIP), "install", "--quiet", "--upgrade", "pip"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if (Path(__file__).parent / "requirements.txt").exists():
            subprocess.check_call(
                [str(_VENV_PIP), "install", "--quiet", "-r", "requirements.txt"],
            )
        else:
            subprocess.check_call(
                [str(_VENV_PIP), "install", "--quiet", *_PACKAGES],
            )
        _MARKER.touch()
        _notify("StemForge · Setup complete — launching…")

    # ── Re-exec this script inside the managed venv (replaces this process) ──
    os.execv(str(_VENV_PY), [str(_VENV_PY)] + sys.argv)


def _notify(msg: str):
    """Show a macOS notification banner (non-blocking, best-effort)."""
    try:
        subprocess.Popen(
            ["osascript", "-e",
             f'display notification "{msg}" with title "StemForge"'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        print(msg)   # fallback: just print if osascript unavailable


_bootstrap()
# ─────────────────────────────────────────────────────────────────────────────

import platform, json, queue, uuid, threading, tempfile, time
from pathlib import Path
from datetime import datetime

from flask import Flask, Response, request, jsonify, stream_with_context


# ── GPU Detection ─────────────────────────────────────────────────────────────
def detect_device():
    try:
        import torch
        if platform.system() == "Darwin" and hasattr(torch.backends, "mps") \
                and torch.backends.mps.is_available():
            return "mps", "Apple Silicon · MPS", 0.0, False
        if torch.cuda.is_available():
            name  = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram  = props.total_memory / (1024 ** 3)
            amp   = props.major == 8          # Ampere = compute capability 8.x
            label = f"{name} · {vram:.1f} GB" + (" · Ampere ✓" if amp else "")
            return "cuda", label, vram, amp
    except ImportError:
        pass
    return "cpu", "CPU  (no GPU)", 0.0, False

def vram_used():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / (1024 ** 3)
    except Exception:
        pass
    return 0.0

def cuda_clear():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

def rec_segment(vram_gb, hint, max_seg=7):
    """Optimal --segment value for available VRAM, capped by the model's architecture limit."""
    usable = max(vram_gb - 1.0, 0.5)
    vram_based = max(5, min(int((usable / hint) * 7), 30))
    return min(vram_based, max_seg)   # never exceed the model's hard cap


DEVICE, DEVICE_LABEL, VRAM_GB, IS_AMPERE = detect_device()

MODELS = {
    "htdemucs": {
        "name": "HTDemucs 4-stem", "badge": "Recommended",
        "stems": ["drums", "bass", "vocals", "other"],
        "desc": "Best all-round quality. Fast on CUDA.",
        "vram_hint": 2.0,
        "max_seg": 7,   # Transformer context window: trained on 7.8 s chunks
    },
    "htdemucs_6s": {
        "name": "HTDemucs 6-stem", "badge": "Extended",
        "stems": ["drums", "bass", "vocals", "other", "guitar", "piano"],
        "desc": "Separates guitar & piano individually.",
        "vram_hint": 3.5,
        "max_seg": 7,   # Strict hard limit — demucs will fatal-error above 7.8 s
    },
    "htdemucs_ft": {
        "name": "HTDemucs Fine-Tuned", "badge": "Vocals",
        "stems": ["drums", "bass", "vocals", "other"],
        "desc": "Optimised for clean vocal isolation.",
        "vram_hint": 2.0,
        "max_seg": 7,   # Same transformer architecture as htdemucs
    },
    "mdx_extra": {
        "name": "MDX-Net", "badge": "Fastest",
        "stems": ["vocals", "no_vocals"],
        "desc": "Quick vocal removal. Minimum VRAM.",
        "vram_hint": 1.5,
        "max_seg": 30,  # CNN-based — no transformer context limit
    },
}

AUDIO_EXT = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".ogg", ".m4a", ".opus"}


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB

_jobs: dict = {}
_jlock = threading.Lock()


@app.route("/")
def index():
    return HTML


@app.route("/api/init")
def api_init():
    seg = rec_segment(VRAM_GB, 2.0, max_seg=7)
    has_df = False
    try:
        import sys, types
        if 'torchaudio.backend' not in sys.modules:
            sys.modules['torchaudio.backend'] = types.ModuleType('torchaudio.backend')
            sys.modules['torchaudio.backend.common'] = types.ModuleType('torchaudio.backend.common')
            sys.modules['torchaudio.backend.common'].AudioMetaData = type('AudioMetaData', (), {})
        import df as _  # deepfilternet
        has_df = True
    except ImportError:
        pass
    return jsonify({
        "device":         DEVICE,
        "device_label":   DEVICE_LABEL,
        "vram_gb":        VRAM_GB,
        "is_ampere":      IS_AMPERE,
        "models":         MODELS,
        "rec_segment":    seg,
        "default_fp16":   IS_AMPERE and DEVICE == "cuda",
        "default_out":    str(Path.home() / "Desktop" / "StemForge Output"),
        "has_deepfilter": has_df,
    })


@app.route("/api/vram")
def api_vram():
    used = vram_used()
    return jsonify({"used": used, "total": VRAM_GB})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f   = request.files["file"]
    suf = Path(f.filename or "").suffix.lower()
    if suf not in AUDIO_EXT:
        return jsonify({"error": f"Unsupported format: {suf}"}), 400
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
    f.save(tmp.name); tmp.close()
    size = os.path.getsize(tmp.name) / (1024 * 1024)
    return jsonify({"path": tmp.name, "name": f.filename, "size_mb": round(size, 1)})


@app.route("/api/separate", methods=["POST"])
def api_separate():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    job_id = str(uuid.uuid4())[:8]
    q = queue.Queue()
    with _jlock:
        _jobs[job_id] = {"queue": q, "status": "running", "out": None}
    threading.Thread(target=_separate, args=(job_id, q, data), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/progress/<job_id>")
def api_progress(job_id):
    with _jlock:
        if job_id not in _jobs:
            return jsonify({"error": "Not found"}), 404
        q = _jobs[job_id]["queue"]

    def generate():
        while True:
            try:
                msg = q.get(timeout=30)
                if msg is None:
                    yield f"data: {json.dumps({'type':'done'})}\n\n"
                    break
                yield f"data: {json.dumps(msg)}\n\n"
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/job/<job_id>")
def api_job(job_id):
    with _jlock:
        if job_id not in _jobs:
            return jsonify({"error": "Not found"}), 404
        j = _jobs[job_id]
        return jsonify({"status": j["status"], "output_path": j.get("out")})


@app.route("/api/open-folder", methods=["POST"])
def api_open_folder():
    path = (request.get_json() or {}).get("path", "")
    if not path or not os.path.exists(path):
        return jsonify({"error": "Not found"}), 400
    try:
        sys_name = platform.system()
        if sys_name == "Darwin":    subprocess.run(["open",     path])
        elif sys_name == "Windows": os.startfile(path)
        else:                       subprocess.run(["xdg-open", path])
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ── Demucs patched launcher ────────────────────────────────────────────────────
# Written to a temp .py file at runtime; patches torchaudio before demucs runs.
_DEMUCS_WRAPPER = """
import sys
sys.argv = __ARGV__

import numpy as _np, wave as _wv

def _load(uri, frame_offset=0, num_frames=-1, normalize=True,
           channels_first=True, format=None, backend=None):
    import torch
    with _wv.open(str(uri), "rb") as f:
        ch, sw, sr = f.getnchannels(), f.getsampwidth(), f.getframerate()
        raw = f.readframes(f.getnframes())
    dt = {1: _np.uint8, 2: _np.int16, 4: _np.int32}[sw]
    a = _np.frombuffer(raw, dtype=dt).reshape(-1, ch).copy().astype(_np.float32)
    if sw == 1:   a = (a - 128.0) / 128.0
    elif sw == 2: a /= 32768.0
    elif sw == 4: a /= 2147483648.0
    t = torch.from_numpy(a)
    return (t.T if channels_first else t), sr

def _save(uri, src, sample_rate, channels_first=True, format=None,
           encoding=None, bits_per_sample=None, buffer_size=4096,
           backend=None, compression=None):
    audio = (src if channels_first else src.T).T.cpu().numpy()
    pcm = (audio.clip(-1.0, 1.0) * 32767.0).astype(_np.int16)
    ch = pcm.shape[1] if pcm.ndim == 2 else 1
    with _wv.open(str(uri), "wb") as f:
        f.setnchannels(ch)
        f.setsampwidth(2)
        f.setframerate(int(sample_rate))
        f.writeframes(pcm.tobytes())

import torchaudio as _ta
_ta.load = _load
_ta.save = _save

from demucs.__main__ import main
try:
    main()
except SystemExit:
    pass   # demucs calls sys.exit(0) — swallow it so DF post-processing can run

# ── DeepFilterNet post-processing ─────────────────────────────────────────────
_DF_ENABLED = __DF_ENABLED__
_DF_ATTEN   = __DF_ATTEN__

if _DF_ENABLED:
    import os as _os, glob as _gl
    _out_dir = sys.argv[sys.argv.index("--out") + 1]
    _stems   = sorted(_gl.glob(_os.path.join(_out_dir, "**", "*.wav"), recursive=True))
    _total   = len(_stems)
    if _total == 0:
        print("[DF] no-stems")
    else:
        print(f"[DF] start {_total}")
        try:
            import sys, types
            if 'torchaudio.backend' not in sys.modules:
                sys.modules['torchaudio.backend'] = types.ModuleType('torchaudio.backend')
                sys.modules['torchaudio.backend.common'] = types.ModuleType('torchaudio.backend.common')
                sys.modules['torchaudio.backend.common'].AudioMetaData = type('AudioMetaData', (), {})
            import numpy as _np, soundfile as _sf, torch as _torch
            from df.enhance import enhance, init_df
            _model, _df_state, _ = init_df()
            try: _sr = _df_state.sr()
            except TypeError: _sr = _df_state.sr
            for _i, _stem in enumerate(_stems, 1):
                _name = _os.path.basename(_stem)
                print(f"[DF] processing {_i}/{_total} {_name}")
                _data, _file_sr = _sf.read(_stem, always_2d=True)
                if _file_sr != _sr:
                    import resampy as _rsp
                    _data = _rsp.resample(_data, _file_sr, _sr, axis=0)
                _audio = _torch.from_numpy(_data.T.astype(_np.float32))
                _enhanced = enhance(_model, _df_state, _audio, atten_lim_db=_DF_ATTEN)
                _arr = _enhanced.squeeze(0).cpu().numpy()
                _arr = _arr[:, _np.newaxis] if _arr.ndim == 1 else _arr.T
                _sf.write(_stem, _arr, _sr, subtype="PCM_16")
                print(f"[DF] done {_i}/{_total}")
            print("[DF] complete")
        except ImportError as _ie:
            _pkg = "soundfile" if "soundfile" in str(_ie) else "resampy" if "resampy" in str(_ie) else None
            if _pkg: print(f"[DF] error {_pkg} not installed — run: pip install {_pkg}")
            else: print("[DF] not-installed")
        except Exception as _e:
            print(f"[DF] error {_e}")
"""

# ── Audio pre-converter ───────────────────────────────────────────────────────
def _to_pcm_wav(src: str, dest_name: str = "") -> "str | None":
    """
    Re-encode *src* as a 16-bit stereo PCM WAV so torchaudio never needs
    torchcodec.  Three methods tried in order (all use stdlib or demucs deps):
      1. soundfile  — handles WAV/FLAC/OGG/AIFF (pip install soundfile)
      2. wave       — Python stdlib, WAV-only, zero extra deps
      3. ffmpeg     — subprocess call, handles everything including MP3
    Returns the path of the new file, or None if all three methods fail.
    """
    import tempfile, numpy as np
    import os as _os
    base = dest_name or _os.path.splitext(_os.path.basename(src))[0]
    out  = _os.path.join(_os.path.dirname(src), base + "_pcm.wav")

    # ── Method 1: soundfile ───────────────────────────────────────────────────
    try:
        import soundfile as sf
        data, sr = sf.read(src, always_2d=True)
        sf.write(out, data, sr, subtype="PCM_16")
        return out
    except ImportError:
        pass
    except Exception:
        pass

    # ── Method 2: stdlib wave (WAV input only, no extra deps) ─────────────────
    try:
        import wave
        with wave.open(src, "rb") as wf:
            ch, sw, sr, nf = (wf.getnchannels(), wf.getsampwidth(),
                              wf.getframerate(), wf.getnframes())
            raw = wf.readframes(nf)
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw)
        if dtype is None:
            raise ValueError(f"Unsupported sample width {sw}")
        audio = np.frombuffer(raw, dtype=dtype).reshape(-1, ch)
        if sw == 1:       # uint8 → int16
            audio = ((audio.astype(np.float32) - 128.0) / 128.0 * 32767).astype(np.int16)
        elif sw == 4:     # int32 → int16
            audio = (audio.astype(np.float64) / 2**31 * 32767).astype(np.int16)
        with wave.open(out, "wb") as wf2:
            wf2.setnchannels(ch)
            wf2.setsampwidth(2)
            wf2.setframerate(sr)
            wf2.writeframes(audio.tobytes())
        return out
    except Exception:
        pass

    # ── Method 3: ffmpeg subprocess ───────────────────────────────────────────
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ar", "44100", "-ac", "2",
             "-sample_fmt", "s16", "-f", "wav", out],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0:
            return out
    except Exception:
        pass

    return None   # all methods failed


# ── Separation worker ─────────────────────────────────────────────────────────
def _separate(job_id: str, q: queue.Queue, data: dict):
    emit  = q.put
    log   = lambda t, lv="info": emit({"type": "log",      "text": t, "level": lv})
    prog  = lambda p, s:         emit({"type": "progress", "pct":  p, "status": s})

    try:
        inp    = data["file_path"]
        mdl    = data["model_id"]
        outdir = data.get("output_dir") or str(Path.home() / "Desktop" / "StemForge Output")
        seg    = int(data.get("segment", 10))
        ov     = float(data.get("overlap", 0.10))
        sh     = int(data.get("shifts", 0))
        jbs    = int(data.get("jobs", 1))
        fp16   = bool(data.get("fp16", IS_AMPERE)) and DEVICE == "cuda"
        df_on  = bool(data.get("deep_filter", False))
        df_att = int(data.get("df_atten", 20))

        cuda_clear()

        # ── Pre-convert to plain PCM WAV (zero extra deps) ──────────────────────
        # torchaudio >= 2.5 routes all loads through torchcodec, which is usually
        # not installed. We rewrite the audio as a standard PCM WAV before demucs
        # sees it. Three methods tried: soundfile → built-in wave → ffmpeg.
        original_stem = Path(data["file_path"]).stem  # e.g. "True Colours"
        pcm_path = _to_pcm_wav(inp, dest_name=original_stem)
        if pcm_path:
            log(f"Pre-converted to PCM WAV: {Path(pcm_path).name}", "info")
            inp = pcm_path
        else:
            log("Could not pre-convert audio — trying original file (may fail)", "warn")

        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(outdir) / f"{Path(inp).stem}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        log(f"■ Model   {mdl}")
        log(f"■ Device  {DEVICE}  segment={seg}s  overlap={ov}  fp16={fp16}")
        log(f"■ Shifts  {sh}  workers={jbs}")
        log(f"■ Output  {run_dir}")
        log("Launching demucs — stand by…")
        prog(5, "Initialising model…")

        # Build demucs argument list, then write a temp wrapper script that
        # patches torchaudio before calling demucs.main() — this bypasses the
        # torchcodec requirement introduced in torchaudio >= 2.5.
        demucs_args = [
            "--name",    mdl,
            "--device",  DEVICE,
            "--out",     str(run_dir),
            "--segment", str(seg),
            "--overlap", str(ov),
            "--shifts",  str(sh),
            "--jobs",    str(jbs),
        ]
        if not fp16:
            demucs_args.append("--float32")
        demucs_args.append(inp)

        import tempfile as _tf
        wrapper_script = (
            _DEMUCS_WRAPPER
            .replace("__ARGV__",       json.dumps(["demucs"] + demucs_args))
            .replace("__DF_ENABLED__", "True" if df_on  else "False")
            .replace("__DF_ATTEN__",   str(df_att))
        )
        wrapper_f = _tf.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        )
        wrapper_f.write(wrapper_script)
        wrapper_f.close()
        wrapper_path = wrapper_f.name

        env = os.environ.copy()
        env['STEMFORGE_SUBPROCESS'] = '1'

        if getattr(sys, 'frozen', False):
            cmd = [sys.executable, '--internal-exec', wrapper_path]
        else:
            cmd = [sys.executable, wrapper_path]
            
        out_f = _tf.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=subprocess.STDOUT, env=env)
        
        pct = 10
        with open(out_f.name, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    if proc.poll() is not None:
                        for rest in f:
                            rest = rest.strip()
                            if rest:
                                log(rest)
                        break
                    time.sleep(0.05)
                    continue
                
                line = line.strip()
                if not line:
                    continue
                if line.startswith("[DF]"):
                    tag = line[5:]
                    if tag.startswith("start"):
                        df_total = int(tag.split()[1])
                        log(f"DeepFilterNet — processing {df_total} stem(s)…", "info")
                        prog(92, "DeepFilterNet starting…")
                    elif tag.startswith("processing"):
                        parts = tag.split()
                        i, tot = map(int, parts[1].split("/"))
                        log(f"DeepFilter {parts[1]}: {parts[2]}", "info")
                        prog(92 + (i / tot) * 7, f"DeepFilter {parts[1]}…")
                    elif tag == "complete":
                        log("DeepFilterNet complete!", "ok")
                        prog(99, "DeepFilter done")
                    elif tag == "not-installed":
                        log("DeepFilterNet not installed — run: pip install deepfilternet", "warn")
                    elif tag.startswith("error"):
                        log(f"DeepFilter error: {tag[6:]}", "err")
                    continue
                log(line)
                if "%" in line:
                    try:
                        raw = float(line.split("%")[0].split()[-1])
                        pct = 10 + raw * (0.80 if df_on else 0.85)
                        prog(pct, f"Separating… {int(raw)}%")
                    except Exception:
                        pass

        if proc.returncode == 0:
            with _jlock:
                _jobs[job_id].update(status="done", out=str(run_dir))
            prog(100, "Complete!")
            log("Separation complete!", "ok")
            if df_on:
                log("Stems processed with DeepFilterNet", "ok")
            log(f"Output: {run_dir}", "ok")
            emit({"type": "success", "output_path": str(run_dir)})
            time.sleep(0.3)   # ensure SSE flushes success before sentinel
        else:
            raise RuntimeError("demucs exited with an error — see log above.")

    except FileNotFoundError:
        log("demucs not found.  Run:  pip install demucs", "err")
        emit({"type": "error", "text": "demucs not found — run: pip install demucs"})
    except Exception as e:
        log(f"ERROR: {e}", "err")
        emit({"type": "error", "text": str(e)})
    finally:
        cuda_clear()
        q.put(None)
        for tmp in [data.get("file_path"),
                    pcm_path if "pcm_path" in dir() else None,
                    wrapper_path if "wrapper_path" in dir() else None]:
            try:
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)
            except Exception:
                pass


# ── Embedded single-page app ──────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>StemForge</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=JetBrains+Mono:ital,wght@0,300;0,400;0,600;1,300&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#03030a;--surf:#07070f;--card:#0b0b16;
  --b0:#252848;--b1:#343870;
  --g:#39ff14;--gd:rgba(57,255,20,.13);--gg:0 0 14px rgba(57,255,20,.4),0 0 36px rgba(57,255,20,.12);
  --c:#00d4ff;--cd:rgba(0,212,255,.15);
  --txt:#a8bfd4;--hi:#e4eef8;--mu:#6a8298;
  --ok:#39ff14;--er:#ff4455;--wa:#ffaa00;
  --r:8px;
}
html,body{height:100%;font-family:'JetBrains Mono',monospace;background:var(--bg);color:var(--txt);overflow:hidden}
/* scanlines */
body::after{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.07) 2px,rgba(0,0,0,.07) 4px);pointer-events:none;z-index:9999}
input,select{font-family:inherit;font-size:12px;background:var(--card);color:var(--hi);border:1px solid var(--b0);border-radius:var(--r);padding:8px 12px;outline:none;transition:border-color .2s}
input:focus{border-color:var(--g);box-shadow:0 0 0 2px var(--gd)}
input[type=range],input[type=checkbox],input[type=radio]{background:none;border:none;box-shadow:none;padding:0}

/* ── layout ── */
#app{display:flex;flex-direction:column;height:100vh}
.hdr{height:54px;background:var(--surf);border-bottom:1px solid var(--b0);display:flex;align-items:center;justify-content:space-between;padding:0 22px;flex-shrink:0}
.logo-mark{font-family:'Orbitron',sans-serif;font-weight:900;font-size:17px;color:var(--g);letter-spacing:5px;text-shadow:var(--gg)}
.logo-sub{font-size:9px;color:#7e9ab4;letter-spacing:3px;margin-left:12px;text-transform:uppercase;align-self:flex-end;padding-bottom:2px}
.hdr-right{display:flex;align-items:center;gap:14px}
.badge{padding:3px 10px;border-radius:4px;font-size:9px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;border:1px solid}
.badge-cuda{background:rgba(118,185,0,.15);color:#76b900;border-color:rgba(118,185,0,.3)}
.badge-mps{background:var(--cd);color:var(--c);border-color:rgba(0,212,255,.25)}
.badge-cpu{background:var(--mu);color:var(--txt);border-color:var(--b0)}
.vram-wrap{display:none;align-items:center;gap:8px;font-size:9px;color:#7e9ab4}
.vram-track{width:72px;height:3px;background:var(--b0);border-radius:2px;overflow:hidden}
.vram-fill{height:100%;background:var(--g);box-shadow:0 0 6px var(--g);transition:width .6s;width:0%}

.main{display:flex;flex:1;overflow:hidden}
.left{width:408px;flex-shrink:0;overflow-y:auto;overflow-x:hidden;padding:18px 14px 100px 18px;border-right:1px solid var(--b0)}
.left::-webkit-scrollbar{width:3px}
.left::-webkit-scrollbar-thumb{background:var(--b0);border-radius:2px}
.right{flex:1;display:flex;flex-direction:column;overflow:hidden;padding:18px 18px 100px 14px;gap:10px}

/* ── cards ── */
.sec-label{font-size:9px;font-weight:600;letter-spacing:3px;color:#7e9ab4;text-transform:uppercase;margin:16px 0 6px 2px}
.sec-label:first-child{margin-top:4px}
.card{background:var(--card);border:1px solid var(--b0);border-radius:var(--r);transition:border-color .2s;overflow:hidden}
.card:hover{border-color:var(--b1)}
.cb{padding:12px 13px 14px}

/* ── drop zone ── */
.dz{border:2px dashed var(--b0);border-radius:6px;padding:18px 12px 14px;text-align:center;cursor:pointer;transition:border-color .25s,background .25s;position:relative}
.dz:hover,.dz.over{border-color:var(--g);background:var(--gd)}
.dz input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.dz-icon{font-size:26px;margin-bottom:7px;display:block;filter:drop-shadow(0 0 8px var(--g))}
.dz-txt{font-size:11px;color:var(--txt)}
.dz-hint{font-size:9px;color:#7e9ab4;margin-top:3px;opacity:.85}
#waveform-wrap{display:none;margin-top:10px;background:rgba(0,0,0,.45);border-radius:4px;padding:5px 0 4px}
#waveform{width:100%;height:50px;display:block}
#file-info{display:none;font-size:11px;color:var(--ok);margin-top:8px}

/* ── model grid ── */
.mgrid{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.mc{border:1px solid var(--b0);border-radius:6px;padding:10px;cursor:pointer;transition:border-color .2s,background .2s}
.mc:hover{border-color:var(--b1);background:rgba(255,255,255,.02)}
.mc.active{border-color:var(--g);background:var(--gd)}
.mc-name{font-size:11px;font-weight:600;color:var(--hi)}
.mc-badge{display:inline-block;font-size:8px;padding:1px 5px;border-radius:3px;background:var(--cd);color:var(--c);margin-left:5px;vertical-align:middle}
.mc-desc{font-size:9px;color:#7e9ab4;margin-top:3px}
.stems{display:flex;flex-wrap:wrap;gap:4px;margin-top:7px}
.sp{font-size:8px;padding:2px 6px;border-radius:3px;background:var(--gd);color:var(--g);border:1px solid rgba(57,255,20,.18)}

/* ── GPU sliders ── */
.pgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
.pg label{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#7e9ab4;display:flex;justify-content:space-between;align-items:center;margin-bottom:7px}
.pv{color:var(--g);font-weight:600;font-size:13px}
input[type=range]{-webkit-appearance:none;width:100%;height:3px;border-radius:2px;background:var(--b0);cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:13px;height:13px;border-radius:50%;background:var(--g);box-shadow:0 0 7px var(--g);cursor:pointer;transition:transform .15s}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.4)}

/* ── toggles ── */
.trow{display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-top:1px solid var(--b0);font-size:11px}
.tlbl{color:var(--hi)}
.tsub{font-size:9px;color:#7e9ab4;display:block;margin-top:2px}
.tog{position:relative;width:34px;height:19px;flex-shrink:0}
.tog input{display:none}
.tog-track{position:absolute;inset:0;border-radius:10px;background:var(--b0);cursor:pointer;transition:background .2s}
.tog input:checked~.tog-track{background:var(--g);box-shadow:0 0 8px rgba(57,255,20,.4)}
.tog-thumb{position:absolute;top:3px;left:3px;width:13px;height:13px;border-radius:50%;background:#fff;pointer-events:none;transition:transform .2s}
.tog input:checked~.tog-thumb{transform:translateX(15px)}

/* ── log ── */
.log-hdr{display:flex;align-items:center;justify-content:space-between}
.log-ttl{font-size:9px;font-weight:600;letter-spacing:3px;color:#7e9ab4;text-transform:uppercase}
.log-clr{font-size:9px;color:#7e9ab4;cursor:pointer;padding:2px 7px;border:1px solid var(--b0);border-radius:3px;background:none;font-family:inherit;transition:color .2s,border-color .2s}
.log-clr:hover{color:var(--hi);border-color:var(--b1)}
#log{flex:1;background:var(--surf);border:1px solid var(--b0);border-radius:var(--r);overflow-y:auto;padding:12px;font-size:11px;line-height:1.75;min-height:0}
#log::-webkit-scrollbar{width:3px}
#log::-webkit-scrollbar-thumb{background:var(--b0)}
.ll{white-space:pre-wrap;word-break:break-all}
.ll.ok{color:var(--ok)}
.ll.err{color:var(--er)}
.ll.warn{color:var(--wa)}
.ll.info{color:var(--txt)}
.lts{color:#7e9ab4;margin-right:8px}
.cur{display:inline-block;width:6px;height:11px;background:var(--g);margin-left:3px;vertical-align:middle;animation:blink 1s step-end infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

/* ── success ── */
#sbanner{display:none;background:rgba(57,255,20,.08);border:1px solid rgba(57,255,20,.25);border-radius:var(--r);padding:10px 14px}
#sbanner p{font-size:11px;color:var(--ok)}
#sbanner button{margin-top:8px;padding:5px 13px;background:var(--gd);color:var(--ok);border:1px solid rgba(57,255,20,.25);border-radius:4px;font-family:inherit;font-size:10px;cursor:pointer;transition:background .2s}
#sbanner button:hover{background:rgba(57,255,20,.18)}

/* ── footer ── */
.ftr{position:fixed;bottom:0;left:0;right:0;height:70px;background:var(--surf);border-top:1px solid var(--b0);display:flex;align-items:center;padding:0 22px;gap:16px;z-index:100}
.ftr-l{flex:1;min-width:0}
#stxt{font-size:11px;color:#7e9ab4;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#stxt.ok{color:var(--ok)}
#stxt.er{color:var(--er)}
.ptrack{height:3px;background:var(--b0);border-radius:2px;overflow:hidden}
#pfill{height:100%;background:var(--g);border-radius:2px;width:0%;transition:width .4s;box-shadow:0 0 7px var(--g)}
#pfill.pulse{animation:pglow 1.4s ease-in-out infinite}
@keyframes pglow{0%,100%{box-shadow:0 0 6px var(--g)}50%{box-shadow:0 0 20px var(--g),0 0 40px rgba(57,255,20,.2)}}
#runbtn{padding:0 26px;height:42px;background:var(--g);color:#040c02;border:none;border-radius:6px;font-family:'Orbitron',sans-serif;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;cursor:pointer;flex-shrink:0;box-shadow:0 0 18px rgba(57,255,20,.3);transition:box-shadow .2s,opacity .2s}
#runbtn:hover:not(:disabled){box-shadow:0 0 28px rgba(57,255,20,.55)}
#runbtn:disabled{opacity:.35;cursor:not-allowed}
#runbtn.busy{background:var(--b0);color:var(--mu);box-shadow:none}
#autotune{cursor:pointer;color:var(--g);font-size:9px;margin-left:10px;opacity:.8;transition:opacity .2s;text-shadow:0 0 8px var(--g)}
#autotune:hover{opacity:1}
</style>
</head>
<body>
<div id="app">

<header class="hdr">
  <div style="display:flex;align-items:flex-end">
    <span class="logo-mark">STEMFORGE</span>
    <span class="logo-sub">AI Stem Separator</span>
  </div>
  <div class="hdr-right">
    <div class="vram-wrap" id="vwrap">
      <span id="vtxt">VRAM 0.00 / 0 GB</span>
      <div class="vram-track"><div class="vram-fill" id="vfill"></div></div>
    </div>
    <span class="badge" id="dbadge">Detecting…</span>
  </div>
</header>

<div class="main">

  <!-- ── Left: settings ── -->
  <div class="left">

    <div class="sec-label">01 — Audio File</div>
    <div class="card">
      <div class="cb">
        <div class="dz" id="dz">
          <input type="file" id="fi" accept=".mp3,.wav,.flac,.aiff,.aif,.ogg,.m4a,.opus">
          <span class="dz-icon">⬡</span>
          <div class="dz-txt">Drop audio file here or click to browse</div>
          <div class="dz-hint">MP3 · WAV · FLAC · AIFF · OGG · M4A · OPUS</div>
        </div>
        <div id="waveform-wrap"><canvas id="waveform" height="50"></canvas></div>
        <div id="file-info"></div>
      </div>
    </div>

    <div class="sec-label">02 — Separation Model</div>
    <div class="card">
      <div class="cb">
        <div class="mgrid" id="mgrid"></div>
      </div>
    </div>

    <div class="sec-label">03 — GPU Settings <span id="autotune">⚡ AUTO-TUNE</span></div>
    <div class="card">
      <div class="cb">
        <div class="pgrid">
          <div class="pg">
            <label>Segment (s)<span class="pv" id="sv">10</span></label>
            <input type="range" id="seg" min="5" max="30" step="1" value="7">
          </div>
          <div class="pg">
            <label>Overlap<span class="pv" id="ov">0.10</span></label>
            <input type="range" id="ovl" min="0.05" max="0.25" step="0.05" value="0.10">
          </div>
        </div>
        <div class="trow">
          <div>
            <span class="tlbl">FP16 Half-Precision</span>
            <span class="tsub">Ampere Tensor Cores — ~2× faster, half the VRAM</span>
          </div>
          <label class="tog">
            <input type="checkbox" id="fp16" checked>
            <div class="tog-track"></div><div class="tog-thumb"></div>
          </label>
        </div>
        <div class="trow">
          <div>
            <span class="tlbl">Extra Shifts (quality boost)</span>
            <span class="tsub">+1 shift — higher SDR, roughly 2× slower</span>
          </div>
          <label class="tog">
            <input type="checkbox" id="shifts">
            <div class="tog-track"></div><div class="tog-thumb"></div>
          </label>
        </div>
      </div>
    </div>

    <div class="sec-label">05 — Post-Processing</div>
      <div class="card">
        <div class="cb">
          <div class="trow" style="border-top:none;padding-top:0">
            <div>
              <span class="tlbl">DeepFilterNet artifact removal
                <span id="df-badge" style="display:none;font-size:9px;padding:1px 6px;border-radius:3px;margin-left:6px;background:rgba(57,255,20,.15);color:var(--g);border:1px solid rgba(57,255,20,.2)">INSTALLED</span>
                <span id="df-badge-no" style="display:none;font-size:9px;padding:1px 6px;border-radius:3px;margin-left:6px;background:rgba(255,68,85,.1);color:var(--er);border:1px solid rgba(255,68,85,.2)">pip install deepfilternet</span>
              </span>
              <span class="tsub">Removes metallic smearing &amp; AI artifacts from each stem</span>
            </div>
            <label class="tog">
              <input type="checkbox" id="df-toggle">
              <div class="tog-track"></div><div class="tog-thumb"></div>
            </label>
          </div>
          <div id="df-atten-wrap" style="display:none;margin-top:10px">
            <div class="pg" style="margin-bottom:0">
              <label>Attenuation limit (dB)<span class="pv" id="df-atten-val">20</span></label>
              <input type="range" id="df-atten" min="5" max="40" step="5" value="20">
              <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--mu);margin-top:4px"><span>more aggressive</span><span>more gentle</span></div>
            </div>
          </div>
        </div>
      </div>

    <div class="sec-label">06 — Output Folder</div>
    <div class="card">
      <div class="cb">
        <input type="text" id="outpath" style="width:100%" placeholder="Output folder path…">
      </div>
    </div>

  </div><!-- /left -->

  <!-- ── Right: log ── -->
  <div class="right">
    <div id="sbanner">
      <p id="stxt2"></p>
      <button id="openbtn">📂 Open Output Folder</button>
    </div>
    <div class="log-hdr">
      <span class="log-ttl">07 — Process Log</span>
      <button class="log-clr" id="logclr">CLEAR</button>
    </div>
    <div id="log"></div>
  </div>

</div><!-- /main -->

<footer class="ftr">
  <div class="ftr-l">
    <div id="stxt">Ready — select an audio file to begin</div>
    <div class="ptrack"><div id="pfill"></div></div>
  </div>
  <button id="runbtn" disabled>SEPARATE</button>
</footer>

</div><!-- /app -->

<script>
(function(){
'use strict';

// ── state ──────────────────────────────────────────────────────────────────
const S={filePath:null,modelId:'htdemucs',seg:10,ovl:.10,shifts:0,fp16:true,
         dfOn:false,dfAtten:20,
         device:'cpu',vramGb:0,running:false,outPath:null,outputDir:''};

// ── refs ───────────────────────────────────────────────────────────────────
const $=id=>document.getElementById(id);
const logEl=$('log'), runBtn=$('runbtn'), stxt=$('stxt'), pfill=$('pfill');
const segR=$('seg'), ovlR=$('ovl'), fp16=$('fp16'), shiftsT=$('shifts');
const fi=$('fi'), dz=$('dz'), outpath=$('outpath');

// ── init ───────────────────────────────────────────────────────────────────
async function init(){
  try{
    const d=await fetch('/api/init').then(r=>r.json());
    S.device=d.device; S.vramGb=d.vram_gb; S.seg=d.rec_segment;
    S.fp16=d.default_fp16; S.outputDir=d.default_out;
    segR.value=d.rec_segment; $('sv').textContent=d.rec_segment;
    fp16.checked=d.default_fp16;
    if(!d.default_fp16||d.device!=='cuda') fp16.disabled=true;
    outpath.value=d.default_out;

    const badge=$('dbadge');
    badge.textContent=d.device_label;
    badge.className='badge '+(d.device==='cuda'?'badge-cuda':d.device==='mps'?'badge-mps':'badge-cpu');
    if(d.device==='cuda'&&d.vram_gb>0){$('vwrap').style.display='flex';pollVram();}
    if(d.has_deepfilter){
      $('df-badge').style.display='inline';
      $('df-toggle').disabled=false;
    } else {
      $('df-badge-no').style.display='inline';
      $('df-toggle').disabled=true;
    }

    buildModels(d.models);
    log2('StemForge ready.','info');
    if(d.is_ampere) log2(`RTX Ampere · ${d.vram_gb.toFixed(1)} GB VRAM · FP16 enabled`,'ok');
  }catch(e){log2('Init error: '+e,'err');}
}

// ── model cards ───────────────────────────────────────────────────────────
function buildModels(models){
  const g=$('mgrid'); g.innerHTML='';
  let first=true;
  for(const[id,m] of Object.entries(models)){
    const d=document.createElement('div');
    d.className='mc'+(first?' active':''); d.dataset.id=id;
    d.dataset.hint=m.vram_hint; d.dataset.maxseg=m.max_seg||7;
    d.innerHTML=`<div class="mc-name">${m.name}<span class="mc-badge">${m.badge}</span></div>`+
      `<div class="mc-desc">${m.desc}</div>`+
      `<div class="stems">${m.stems.map(s=>`<span class="sp">${s}</span>`).join('')}</div>`;
    d.onclick=()=>{
      document.querySelectorAll('.mc').forEach(c=>c.classList.remove('active'));
      d.classList.add('active'); S.modelId=id;
      const seg=recSeg(S.vramGb,+d.dataset.hint,+d.dataset.maxseg);
        segR.max=d.dataset.maxseg||7;
      segR.value=seg; $('sv').textContent=seg; S.seg=seg;
    };
    g.appendChild(d); first=false;
  }
}
function recSeg(v,h,maxSeg){const u=Math.max(v-1,.5);const vramBased=Math.max(5,Math.min(Math.floor(u/h*7),30));return Math.min(vramBased,maxSeg||7);}

// ── vram poll ─────────────────────────────────────────────────────────────
async function pollVram(){
  try{
    const d=await fetch('/api/vram').then(r=>r.json());
    const p=d.total>0?d.used/d.total*100:0;
    $('vfill').style.width=p+'%';
    $('vtxt').textContent=`VRAM ${d.used.toFixed(2)} / ${d.total.toFixed(1)} GB`;
  }catch(_){}
  setTimeout(pollVram,S.running?2000:6000);
}

// ── drag & drop ───────────────────────────────────────────────────────────
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('over');});
dz.addEventListener('dragleave',()=>dz.classList.remove('over'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('over');
  if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);});
fi.addEventListener('change',()=>{if(fi.files[0]) handleFile(fi.files[0]);});

async function handleFile(file){
  log2(`Uploading: ${file.name} (${(file.size/1048576).toFixed(1)} MB)…`,'info');
  const fd=new FormData(); fd.append('file',file);
  try{
    const res=await fetch('/api/upload',{method:'POST',body:fd});
    const d=await res.json();
    if(d.error){log2('Upload error: '+d.error,'err');return;}
    S.filePath=d.path;
    const fi2=$('file-info');
    fi2.textContent=`✓  ${d.name}  ·  ${d.size_mb} MB`; fi2.style.display='block';
    runBtn.disabled=false;
    log2(`File ready: ${d.name}`,'ok');
    drawWaveform(file);
  }catch(e){log2('Upload failed: '+e,'err');}
}

// ── waveform ──────────────────────────────────────────────────────────────
async function drawWaveform(file){
  const wrap=$('waveform-wrap'), canvas=$('waveform');
  wrap.style.display='block';
  canvas.width=canvas.offsetWidth||360;
  const ctx=canvas.getContext('2d'), W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);
  try{
    const buf=await file.arrayBuffer();
    const ac=new(window.AudioContext||window.webkitAudioContext)();
    const decoded=await ac.decodeAudioData(buf);
    const data=decoded.getChannelData(0);
    const step=Math.ceil(data.length/W), mid=H/2;
    ctx.fillStyle='rgba(57,255,20,0.04)'; ctx.fillRect(0,0,W,H);
    ctx.strokeStyle='#39ff14'; ctx.lineWidth=1;
    ctx.shadowBlur=5; ctx.shadowColor='#39ff14';
    ctx.beginPath();
    for(let x=0;x<W;x++){
      let mn=1,mx=-1;
      for(let i=0;i<step;i++){const v=data[x*step+i]||0;if(v<mn)mn=v;if(v>mx)mx=v;}
      ctx.moveTo(x+.5,mid+mn*mid*.78); ctx.lineTo(x+.5,mid+mx*mid*.78);
    }
    ctx.stroke(); await ac.close();
  }catch(_){
    ctx.fillStyle='rgba(57,255,20,.4)'; ctx.font='10px JetBrains Mono';
    ctx.fillText('Waveform unavailable for this format',10,H/2+4);
  }
}

// ── slider events ─────────────────────────────────────────────────────────
segR.addEventListener('input',()=>{S.seg=+segR.value;$('sv').textContent=segR.value;});
ovlR.addEventListener('input',()=>{S.ovl=+ovlR.value;$('ov').textContent=(+ovlR.value).toFixed(2);});
fp16.addEventListener('change',()=>{S.fp16=fp16.checked;});
shiftsT.addEventListener('change',()=>{S.shifts=shiftsT.checked?1:0;});
outpath.addEventListener('input',()=>{S.outputDir=outpath.value;});

$('df-toggle').addEventListener('change',()=>{
  S.dfOn=$('df-toggle').checked;
  $('df-atten-wrap').style.display=S.dfOn?'block':'none';
});
$('df-atten').addEventListener('input',()=>{
  S.dfAtten=+$('df-atten').value;
  $('df-atten-val').textContent=$('df-atten').value;
});

$('autotune').addEventListener('click',()=>{
  const activeCard=document.querySelector('.mc.active');
  const maxSeg=activeCard?+activeCard.dataset.maxseg:7;
  const seg=recSeg(S.vramGb||2,2.0,maxSeg);
  segR.value=seg; $('sv').textContent=seg; S.seg=seg;
  ovlR.value='.10'; $('ov').textContent='0.10'; S.ovl=.10;
  shiftsT.checked=false; S.shifts=0;
  if(S.device==='cuda'){fp16.checked=true; S.fp16=true;}
  log2('⚡ Auto-tuned for RTX 3050','ok');
});

// ── log ───────────────────────────────────────────────────────────────────
let cur=null;
function log2(text,level='info'){
  if(cur) cur.remove();
  const ts=new Date().toTimeString().slice(0,8);
  const ic={ok:'✓',err:'✗',warn:'⚠',info:'·'}[level]||'·';
  const s=document.createElement('span');
  s.className='ll '+level;
  s.innerHTML=`<span class="lts">[${ts}]</span>${ic}  ${text}`;
  logEl.appendChild(s); logEl.appendChild(document.createElement('br'));
  cur=document.createElement('span'); cur.className='cur'; logEl.appendChild(cur);
  logEl.scrollTop=logEl.scrollHeight;
}
$('logclr').addEventListener('click',()=>{logEl.innerHTML=''; cur=null;});

// ── separation ────────────────────────────────────────────────────────────
runBtn.addEventListener('click',startSep);
async function startSep(){
  if(S.running||!S.filePath) return;
  S.running=true;
  runBtn.disabled=true; runBtn.classList.add('busy'); runBtn.textContent='PROCESSING…';
  stxt.textContent='Starting…'; stxt.className='';
  pfill.style.width='5%'; pfill.classList.add('pulse');
  $('sbanner').style.display='none';
  try{
    const r=await fetch('/api/separate',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({file_path:S.filePath,model_id:S.modelId,
        segment:S.seg,overlap:S.ovl,shifts:S.shifts,fp16:S.fp16,jobs:1,
        output_dir:S.outputDir||outpath.value,
        deep_filter:S.dfOn,df_atten:S.dfAtten})});
    const d=await r.json();
    if(d.error) throw new Error(d.error);
    listenSSE(d.job_id);
  }catch(e){onErr(e.message);}
}

function listenSSE(jobId){
  const es=new EventSource(`/api/progress/${jobId}`);
  let completed=false;
  es.onmessage=evt=>{
    const d=JSON.parse(evt.data);
    if(d.type==='log') log2(d.text,d.level||'info');
    else if(d.type==='progress'){pfill.style.width=d.pct+'%'; stxt.textContent=d.status;}
    else if(d.type==='success'){completed=true; es.close(); onOk(d.output_path);}
    else if(d.type==='error'){completed=true; es.close(); onErr(d.text);}
    else if(d.type==='done'){es.close(); if(!completed) pollJobResult(jobId);}
  };
  es.onerror=()=>{es.close(); if(!completed) pollJobResult(jobId);};
}
async function pollJobResult(jobId){
  try{
    const r=await fetch(`/api/job/${jobId}`);
    const d=await r.json();
    if(d.status==='done'&&d.output_path) onOk(d.output_path);
    else if(d.status==='running') setTimeout(()=>pollJobResult(jobId), 3000);
    else onErr(d.error||'Processing may have failed — check log.');
  }catch(e){
    setTimeout(()=>pollJobResult(jobId), 3000);
  }
}

function onOk(path){
  S.running=false; S.filePath=null; S.outPath=path;
  pfill.style.width='100%'; pfill.classList.remove('pulse');
  runBtn.disabled=false; runBtn.classList.remove('busy'); runBtn.textContent='SEPARATE';
  stxt.textContent='✓ Complete!'; stxt.className='ok';
  $('stxt2').textContent=`✓  Stems saved to: ${path}`;
  $('sbanner').style.display='block';
  $('openbtn').onclick=()=>fetch('/api/open-folder',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({path})});
}

function onErr(msg){
  S.running=false;
  pfill.style.width='0%'; pfill.classList.remove('pulse');
  runBtn.disabled=false; runBtn.classList.remove('busy'); runBtn.textContent='SEPARATE';
  stxt.textContent='✗ Error — see log'; stxt.className='er';
  log2('ERROR: '+msg,'err');
}

init();
})();
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    import socket
    PORT = 7777
    for p in range(7777, 7850):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('127.0.0.1', p))
                PORT = p
                break
        except socket.error:
            continue

    print(f"""
  ⬡  StemForge Web  ·  RTX 3050 Optimised
  ─────────────────────────────────────────────
  Open in Chrome:  http://localhost:{PORT}
  Press Ctrl+C to stop
""")

    def _open_browser():
        if os.environ.get('STEMFORGE_SUBPROCESS'):
            return                       # don't open browser in subprocess
        time.sleep(1.4)
        import webbrowser
        webbrowser.open(f"http://localhost:{PORT}")

    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
