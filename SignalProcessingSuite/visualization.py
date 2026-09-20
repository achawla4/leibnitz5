"""Plotting helpers for time, frequency, spectrogram, and wavelet views."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .fft_tools import magnitude_spectrum, short_time_fft
    from .utils import validate_1d_signal
except ImportError:
    from fft_tools import magnitude_spectrum, short_time_fft
    from utils import validate_1d_signal


_FONT_PROP = None

def _get_font_prop():
    global _FONT_PROP
    if _FONT_PROP is not None:
        return _FONT_PROP
    try:
        font_path = Path(__file__).resolve().parent.parent / 'fonts' / 'NotoSansDevanagari.ttf'
        if font_path.exists():
            import matplotlib.font_manager as fm
            fm.fontManager.addfont(str(font_path))
            _FONT_PROP = fm.FontProperties(fname=str(font_path))
            return _FONT_PROP
    except Exception:
        pass
    return None


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Matplotlib is required for visualization") from exc
    return plt


def plot_time(signal: Iterable[float], sample_rate: float, ax=None, title: str = "समयक्षेत्रसङ्केतः (Samayakshetra-sanketah)", fontproperties=None):
    """Plot amplitude over time and return the Matplotlib axis."""
    data = validate_1d_signal(signal)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    plt = _pyplot()
    ax = ax or plt.subplots()[1]
    times = np.arange(data.size) / sample_rate
    ax.plot(times, data)
    fp = fontproperties or _get_font_prop()
    if fp:
        ax.set_title(title, fontproperties=fp)
        ax.set_xlabel("कालः / Kalah (s)", fontproperties=fp)
        ax.set_ylabel("आयामः / Ayamah", fontproperties=fp)
    else:
        ax.set(title=title, xlabel="कालः / Kalah (s)", ylabel="आयामः / Ayamah")
    return ax


def plot_frequency(signal: Iterable[float], sample_rate: float, ax=None, db: bool = False, title: str = "आवृत्तिवर्णक्रमः (Avrittivarnakramah FFT)", fontproperties=None):
    """Plot a one-sided magnitude spectrum and return the Matplotlib axis."""
    plt = _pyplot()
    ax = ax or plt.subplots()[1]
    freqs, magnitude = magnitude_spectrum(signal, sample_rate, db=db)
    ax.plot(freqs, magnitude)
    fp = fontproperties or _get_font_prop()
    ylab = "परिमाणम् (dB) / Parimanam (dB)" if db else "परिमाणम् / Parimanam"
    if fp:
        ax.set_title(title, fontproperties=fp)
        ax.set_xlabel("आवृत्तिः / Avrittih (Hz)", fontproperties=fp)
        ax.set_ylabel(ylab, fontproperties=fp)
    else:
        ax.set(title=title, xlabel="आवृत्तिः / Avrittih (Hz)", ylabel=ylab)
    return ax


def plot_ifft(signal: Iterable[float], sample_rate: float, ax=None, db: bool = False, title: str = "समयक्षेत्रे प्रतिलोम-द्रुत-फूर्ये (Samayakshetre IFFT)", fontproperties=None):
    """Plot a timedomain ifft signal and return the Matplotlib axis."""
    plt = _pyplot()
    ax = ax or plt.subplots()[1]
    timesamples = np.real(np.fft.ifft(signal))
    ax.plot(np.arange(len(timesamples)), timesamples)
    fp = fontproperties or _get_font_prop()
    if fp:
        ax.set_title(title, fontproperties=fp)
        ax.set_xlabel("प्रतिचयन-क्रमाङ्कः / Praticayana-kramankah", fontproperties=fp)
        ax.set_ylabel("आयामः / Ayamah", fontproperties=fp)
    else:
        ax.set(title=title, xlabel="प्रतिचयन-क्रमाङ्कः / Praticayana-kramankah", ylabel="आयामः / Ayamah")
    return ax


def plot_spectrogram(
    signal: Iterable[float],
    sample_rate: float,
    frame_size: int = 1024,
    hop_size: int = 256,
    ax=None,
    fontproperties=None,
):
    """Plot a short-time FFT spectrogram and return the Matplotlib axis."""
    plt = _pyplot()
    ax = ax or plt.subplots()[1]
    freqs, times, stft = short_time_fft(signal, sample_rate, frame_size, hop_size)
    image = 20.0 * np.log10(np.maximum(np.abs(stft), np.finfo(float).eps))
    mesh = ax.pcolormesh(times, freqs, image, shading="auto")
    fp = fontproperties or _get_font_prop()
    if fp:
        ax.set_title("वर्णपटचित्रम् (Varnapatacitram Spectrogram)", fontproperties=fp)
        ax.set_xlabel("कालः / Kalah (s)", fontproperties=fp)
        ax.set_ylabel("आवृत्तिः / Avrittih (Hz)", fontproperties=fp)
    else:
        ax.set(title="वर्णपटचित्रम् (Varnapatacitram Spectrogram)", xlabel="कालः / Kalah (s)", ylabel="आवृत्तिः / Avrittih (Hz)")
    cbar = ax.figure.colorbar(mesh, ax=ax)
    if fp:
        cbar.ax.set_ylabel("परिमाणम् (dB) / Parimanam (dB)", fontproperties=fp)
    else:
        cbar.ax.set_ylabel("परिमाणम् (dB) / Parimanam (dB)")
    return ax


def plot_wavelet_coefficients(coeffs: list[np.ndarray], ax=None, fontproperties=None):
    """Plot wavelet coefficient arrays as stacked lines."""
    plt = _pyplot()
    ax = ax or plt.subplots()[1]
    offset = 0.0
    fp = fontproperties or _get_font_prop()
    for index, coeff in enumerate(coeffs):
        arr = np.asarray(coeff, dtype=float)
        scale = np.max(np.abs(arr)) or 1.0
        ax.plot(arr / scale + offset, label=f"स्तरः {index} (Starah {index})")
        offset += 2.0
    if fp:
        ax.set_title("वीचिका-गुणाङ्काः (Vicika-gunankah)", fontproperties=fp)
        ax.set_xlabel("गुणाङ्क-निर्देशाङ्कः / Gunanka-nirdeshankah", fontproperties=fp)
        ax.set_ylabel("प्रसामान्यीकृत-स्तरः / Prasamanyikrita-starah", fontproperties=fp)
        ax.legend(loc="upper right", prop=fp)
    else:
        ax.set(title="वीचिका-गुणाङ्काः (Vicika-gunankah)", xlabel="गुणाङ्क-निर्देशाङ्कः / Gunanka-nirdeshankah", ylabel="प्रसामान्यीकृत-स्तरः / Prasamanyikrita-starah")
        ax.legend(loc="upper right")
    return ax


def show() -> None:
    """Show all pending Matplotlib figures."""
    _pyplot().show()
