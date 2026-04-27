"""
Piano Key Identification Using Frequency Analysis

Uses FFT (Fast Fourier Transform) to identify which piano key was pressed
from a live microphone recording

Installations:
    pip install numpy scipy sounddevice matplotlib

Usage:
    python musicnoteidentifier.py                  # live mic recording
    python piano_key_identifier.py --duration 3     # record for 3 seconds
    python piano_key_identifier.py --list-devices   # show audio input devices
"""

import argparse
import sys
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, find_peaks
import warnings

warnings.filterwarnings("ignore")


# Piano Note Table (MIDI notes 21–108, A0 to C8)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_freq(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz (A4=440 Hz standard)."""
    return 440.0 * (2 ** ((midi_note - 69) / 12))

def midi_to_name(midi_note: int) -> str:
    """Convert MIDI note number to human-readable name, e.g. 69 → 'A4'."""
    octave = (midi_note // 12) - 1
    name = NOTE_NAMES[midi_note % 12]
    return f"{name}{octave}"

# Build full piano key table: MIDI 21 (A0) → 108 (C8)
PIANO_KEYS = [
    {"midi": m, "name": midi_to_name(m), "freq": midi_to_freq(m)}
    for m in range(21, 109)
]


# Signal Processing Helpers


def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float,
                    sample_rate: int, order: int = 5) -> np.ndarray:
    
    nyq = sample_rate / 2.0
    low = lowcut / nyq
    high = min(highcut / nyq, 0.999)  # must be < 1
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, signal)


def compute_fft(signal: np.ndarray, sample_rate: int):
    
    n = len(signal)
    # hann window to reduce spectral leakage
    window = np.hanning(n)
    windowed = signal * window

    spectrum = fft(windowed)
    freqs = fftfreq(n, d=1.0 / sample_rate)

    # keep only positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    magnitudes = np.abs(spectrum[pos_mask])

    return freqs, magnitudes


def find_fundamental(freqs: np.ndarray, magnitudes: np.ndarray,
                     min_freq: float = 27.5, max_freq: float = 4186.0
                     ) -> tuple[float, float]:

    # HPS on full spectrum (indices must start at 0 for correct harmonic alignment)
    hps = magnitudes.copy()
    num_harmonics = 2
    for h in range(2, num_harmonics + 1):
        downsampled = magnitudes[::h]
        min_len = min(len(hps), len(downsampled))
        hps[:min_len] *= downsampled[:min_len]
        hps[min_len:] = 0

    # NOW restrict search to piano range
    mask = (freqs >= min_freq) & (freqs <= max_freq / num_harmonics)
    if not np.any(mask):
        return 0.0, 0.0

    hps_masked = hps.copy()
    hps_masked[~mask] = 0

    
    idx = np.argmax(hps_masked)
    return float(freqs[idx]), float(magnitudes[idx])


def freq_to_piano_key(freq: float) -> dict:
    """
    Match a frequency to the nearest piano key.
    Returns the key dict with an added 'cents_off' field.
    """
    if freq <= 0:
        return {"midi": -1, "name": "?", "freq": 0.0, "cents_off": 0.0}

    best_key = min(PIANO_KEYS, key=lambda k: abs(k["freq"] - freq))

    # Cents deviation: 100 cents = 1 semitone
    cents_off = 1200 * np.log2(freq / best_key["freq"]) if best_key["freq"] > 0 else 0.0

    return {**best_key, "cents_off": round(cents_off, 1)}


# Audio Acquisition


def list_audio_devices():
    """Print all available audio input devices."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print("\n=== Available Audio Input Devices ===")
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                marker = " ← default" if i == sd.default.device[0] else ""
                print(f"  [{i:2d}] {d['name']}  "
                      f"(channels: {d['max_input_channels']}, "
                      f"rate: {int(d['default_samplerate'])} Hz){marker}")
        print()
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)


def record_audio(duration: float = 2.0, sample_rate: int = 44100,
                 device=None) -> np.ndarray:
    """Record audio from the microphone for `duration` seconds."""
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)

    print(f"\n🎹 Recording for {duration:.1f} second(s)... (press a key!)")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="float32", device=device)
    sd.wait()
    print("   Done recording.")
    return audio.flatten()


# Visualization


def plot_spectrum(freqs: np.ndarray, magnitudes: np.ndarray,
                 detected_freq: float, key: dict, title: str = ""):
    """Plot the FFT spectrum and annotate the detected note."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not installed (pip install matplotlib). Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    # Show up to 5 kHz for readability
    mask = freqs <= 5000
    ax.plot(freqs[mask], magnitudes[mask], color="#2196F3", linewidth=0.8,
            label="FFT Magnitude")

    if detected_freq > 0:
        ax.axvline(x=detected_freq, color="#FF5722", linewidth=1.5,
                   linestyle="--", label=f"Detected: {detected_freq:.1f} Hz")
        ax.axvline(x=key["freq"], color="#4CAF50", linewidth=1.5,
                   linestyle=":", label=f"Matched: {key['name']} ({key['freq']:.2f} Hz)")

    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Magnitude", fontsize=11)
    ax.set_title(title or f"FFT Spectrum — Detected Note: {key['name']}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 5000)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


# Core Pipeline


def analyze_signal(signal: np.ndarray, sample_rate: int,
                   show_plot: bool = True) -> dict:
    """
    Full analysis pipeline:
      1. Bandpass filter (piano range: 27.5 Hz – 4200 Hz)
      2. FFT
      3. HPS peak detection → fundamental frequency
      4. Nearest piano key lookup
      5. Optional spectrum plot
    Returns a result dict.
    """
    rms = np.sqrt(np.mean(signal ** 2))
    print(f"\n   Signal RMS level  : {rms:.5f}")

    if rms < 1e-4:
        print("   ⚠  Signal is nearly silent. Check your microphone level.")

    # 1. Filter
    filtered = bandpass_filter(signal, lowcut=27.0, highcut=4200.0,
                                sample_rate=sample_rate)

    # 2. FFT
    freqs, magnitudes = compute_fft(filtered, sample_rate)

    # 3. Fundamental frequency via HPS
    fund_freq, fund_mag = find_fundamental(freqs, magnitudes)

    # 4. Key identification
    key = freq_to_piano_key(fund_freq)

    # 5. Derived info
    freq_resolution = sample_rate / len(signal)

    result = {
        "detected_freq_hz": round(fund_freq, 2),
        "matched_note": key["name"],
        "note_freq_hz": round(key["freq"], 2),
        "cents_off": key["cents_off"],
        "midi_number": key["midi"],
        "freq_resolution_hz": round(freq_resolution, 3),
        "signal_rms": round(rms, 5),
        "sample_rate_hz": sample_rate,
        "signal_length_samples": len(signal),
    }

    # Pretty print
    print("\n" + "═" * 46)
    print("  🎹  PIANO KEY IDENTIFICATION RESULT")
    print("═" * 46)
    print(f"  Detected frequency  : {result['detected_freq_hz']:>10.2f} Hz")
    print(f"  Matched note        : {result['matched_note']:>10s}")
    print(f"  Note frequency      : {result['note_freq_hz']:>10.2f} Hz")
    print(f"  Tuning deviation    : {result['cents_off']:>+10.1f} cents")
    print(f"  MIDI number         : {result['midi_number']:>10d}")
    print(f"  Freq resolution     : {result['freq_resolution_hz']:>10.3f} Hz/bin")
    print("═" * 46)

    tuning_label = "in tune" if abs(key["cents_off"]) < 10 else \
                   ("sharp ▲" if key["cents_off"] > 0 else "flat ▼")
    print(f"\n  → Note is {tuning_label} ({abs(key['cents_off']):.1f} cents off)\n")

    if show_plot:
        plot_spectrum(freqs, magnitudes, fund_freq, key)

    return result


# CLI Entry Point


def main():
    parser = argparse.ArgumentParser(
        description="Identify piano keys from audio using FFT analysis.")
    parser.add_argument("--duration", "-d", type=float, default=2.0,
                        help="Recording duration in seconds (default: 2.0).")
    parser.add_argument("--sample-rate", "-r", type=int, default=44100,
                        help="Sample rate in Hz for live recording (default: 44100).")
    parser.add_argument("--device", type=int, default=None,
                        help="Input device index (see --list-devices).")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio input devices and exit.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable the FFT spectrum plot.")
    parser.add_argument("--loop", "-l", action="store_true",
                        help="Keep recording and identifying until Ctrl+C.")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    show_plot = not args.no_plot

    if args.loop:
        # ── Continuous live recording loop ──
        print("\nRunning in loop mode. Press Ctrl+C to stop.\n")
        try:
            while True:
                signal = record_audio(args.duration, args.sample_rate, args.device)
                analyze_signal(signal, args.sample_rate, show_plot=False)
                print("  (waiting for next keypress…)\n")
        except KeyboardInterrupt:
            print("\nStopped.")

    else:
        # ── Single live recording ──
        signal = record_audio(args.duration, args.sample_rate, args.device)
        analyze_signal(signal, args.sample_rate, show_plot=show_plot)


if __name__ == "__main__":
    main()
