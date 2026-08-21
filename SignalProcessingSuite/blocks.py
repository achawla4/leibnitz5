"""Composable processing blocks for Leibnitz pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable

import numpy as np


try:
    from .fft_tools import magnitude_spectrum
    from .filters import filter_signal
    from .wavelet import dwt
    from .davincitron import MasterDesigner
    from .time_features import MasterAnalyzer	
except ImportError:
    from fft_tools import magnitude_spectrum
    from filters import filter_signal
    from wavelet import dwt
    from davincitron import MasterDesigner
    from time_features import MasterAnalyzer	



@dataclass
class BlockRunResult:
    """Result returned by a processing block."""

    result: dict[str, Any]
    output_signal: np.ndarray | None = None
    plot_data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ProcessingBlock:
    """Base class for registry-backed signal-processing blocks."""

    id = ""
    name = ""
    category = "analysis"
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}
    default_params: dict[str, Any] = {}

    def validate(self, params: dict[str, Any], sample_rate: float) -> dict[str, Any]:
        merged = dict(self.default_params)
        merged.update(params or {})
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        return merged

    def run(self, signal: Iterable[float], sample_rate: float, params: dict[str, Any]) -> BlockRunResult:
        raise NotImplementedError


class FFTBlock(ProcessingBlock):
    id = "fft"
    name = "FFT Analysis"
    category = "frequency"
    input_schema = {"signal": "1d-array", "sample_rate": "positive-float"}
    output_schema = {"frequencies": "array", "magnitude": "array"}
    default_params = {"fft_window": "none"}

    def run(self, signal: Iterable[float], sample_rate: float, params: dict[str, Any]) -> BlockRunResult:
        params = self.validate(params, sample_rate)
        window_type = params.get("fft_window", "none")
        if window_type == "none" or not window_type:
            window_type = None

        freqs, magnitude = magnitude_spectrum(signal, sample_rate=sample_rate, window=window_type)
        return BlockRunResult(
            result={
                "frequencies": freqs.tolist()[:500],
                "magnitude": magnitude.tolist()[:500],
            },
            plot_data={"frequencies": freqs, "magnitude": magnitude},
            metadata={"window": window_type or "none"},
        )

class ifftBlock(ProcessingBlock):
    id = "ifft"
    name = "IFFT Analysis"
    category = "frequency"

    input_schema = {
        "signal": "1d-array",
        "sample_rate": "positive-float"
    }

    output_schema = {
        "timesamples": "array",
        "indices": "array"
    }

    default_params = {
        "ifft_window": "none"
    }

    def run(
        self,
        signal: Iterable[float],
        sample_rate: float,
        params: dict[str, Any]
    ) -> BlockRunResult:

        params = self.validate(params, sample_rate)

        window_type = params.get("ifft_window", "none")
        if window_type == "none" or not window_type:
            window_type = None

        signal = np.asarray(signal)

        timesamples = np.real(np.fft.ifft(signal))
        indices = np.arange(len(timesamples))

        return BlockRunResult(
            result={
                "indices": indices.tolist()[:500],
                "timesamples": timesamples.tolist()[:500],
            },
            plot_data={
                "indices": indices,
                "timesamples": timesamples,
            },
            metadata={
                "window": window_type or "none",
            },
        )
    
class FilterBlock(ProcessingBlock):
    id = "filter"
    name = "Filter"
    category = "conditioning"
    input_schema = {"signal": "1d-array", "sample_rate": "positive-float"}
    output_schema = {"filtered": "array"}
    default_params = {
        "filter_kind": "lowpass",
        "filter_method": "butter",
        "filter_order": 4,
        "filter_cutoff": 100.0,
    }

    def validate(self, params: dict[str, Any], sample_rate: float) -> dict[str, Any]:
        merged = super().validate(params, sample_rate)
        merged["filter_kind"] = merged.get("filter_kind") or "lowpass"
        merged["filter_method"] = merged.get("filter_method") or "butter"
        try:
            merged["filter_order"] = int(merged.get("filter_order", 4))
        except (TypeError, ValueError):
            merged["filter_order"] = 4
        if merged["filter_order"] <= 0:
            raise ValueError("filter_order must be positive")
        merged["cutoff"] = self._parse_cutoff(merged.get("filter_cutoff"), merged["filter_kind"], sample_rate)
        return merged

    def _parse_cutoff(self, raw_cutoff: Any, filter_kind: str, sample_rate: float) -> float | tuple[float, float]:
        nyquist = sample_rate / 2.0
        if filter_kind in ("bandpass", "bandstop"):
            if isinstance(raw_cutoff, (list, tuple)) and len(raw_cutoff) == 2:
                cutoff = (float(raw_cutoff[0]), float(raw_cutoff[1]))
            elif isinstance(raw_cutoff, str) and "," in raw_cutoff:
                parts = raw_cutoff.split(",", 1)
                cutoff = (float(parts[0].strip()), float(parts[1].strip()))
            else:
                cutoff = (0.1 * nyquist, 0.5 * nyquist)
            if not 0 < cutoff[0] < cutoff[1] < nyquist:
                raise ValueError("band cutoff must contain two increasing values below Nyquist")
            return cutoff

        try:
            cutoff_value = float(raw_cutoff) if raw_cutoff is not None else 100.0
        except (TypeError, ValueError):
            cutoff_value = 100.0
        if not 0 < cutoff_value < nyquist:
            raise ValueError("filter_cutoff must be between 0 and Nyquist")
        return cutoff_value

    def run(self, signal: Iterable[float], sample_rate: float, params: dict[str, Any]) -> BlockRunResult:
        params = self.validate(params, sample_rate)
        filtered = filter_signal(
            signal,
            sample_rate=sample_rate,
            cutoff=params["cutoff"],
            kind=params["filter_kind"],
            method=params["filter_method"],
            order=params["filter_order"],
        )
        return BlockRunResult(
            result={"filtered": filtered.tolist()[:1000]},
            output_signal=filtered,
            plot_data={"filtered": filtered},
            metadata={
                "kind": params["filter_kind"],
                "method": params["filter_method"],
                "order": params["filter_order"],
                "cutoff": params["cutoff"],
            },
        )


class WaveletBlock(ProcessingBlock):
    id = "wavelet"
    name = "Wavelet Transform"
    category = "time-frequency"
    input_schema = {"signal": "1d-array", "sample_rate": "positive-float"}
    output_schema = {"coefficients_summary": "array"}
    default_params = {"wavelet_type": "db4", "wavelet_levels": 3}

    def validate(self, params: dict[str, Any], sample_rate: float) -> dict[str, Any]:
        merged = super().validate(params, sample_rate)
        merged["wavelet_type"] = merged.get("wavelet_type") or "db4"
        try:
            merged["wavelet_levels"] = int(merged.get("wavelet_levels", 3))
        except (TypeError, ValueError):
            merged["wavelet_levels"] = 3
        if merged["wavelet_levels"] <= 0:
            raise ValueError("wavelet_levels must be positive")
        return merged

    def run(self, signal: Iterable[float], sample_rate: float, params: dict[str, Any]) -> BlockRunResult:
        params = self.validate(params, sample_rate)
        coeffs = dwt(signal, wavelet=params["wavelet_type"], levels=params["wavelet_levels"])
        summary = [
            {
                "band": "Approximation" if index == 0 else f"Detail Level {len(coeffs) - index}",
                "size": len(coeff),
                "mean": float(np.mean(coeff)),
                "std": float(np.std(coeff)),
                "max": float(np.max(coeff)),
                "min": float(np.min(coeff)),
            }
            for index, coeff in enumerate(coeffs)
        ]
        return BlockRunResult(
            result={
                "wavelet": params["wavelet_type"],
                "levels": len(coeffs) - 1,
                "coefficients_summary": summary,
            },
            plot_data={"_coeffs_obj": coeffs},
            metadata={"wavelet": params["wavelet_type"], "requested_levels": params["wavelet_levels"]},
        )


class DavinciTronBlock(ProcessingBlock):
    id = "davincitron"
    name = "DavinciTron Generator"
    category = "creative"
    input_schema = {"signal": "1d-array", "sample_rate": "positive-float"}
    output_schema = {"canvas": "PIL.Image", "seed": "int", "score": "float", "placed": "int"}
    default_params = {}

    def run(self, signal: Iterable[float], sample_rate: float, params: dict[str, Any]) -> BlockRunResult:
        params = self.validate(params, sample_rate)
        signal_array = np.asarray(signal, dtype=float)
        
        # Calculate seed deterministically from signal
        import hashlib
        signal_bytes = signal_array.tobytes()
        hash_val = hashlib.sha256(signal_bytes).hexdigest()
        seed = int(hash_val, 16) % (2**32)
        
        # Compose canvas using MasterDesigner
        master = MasterDesigner(seed=seed)
        canvas, score, placed = master.compose()
        
        return BlockRunResult(
            result={
                "seed": seed,
                "score": round(score, 4),
                "placed": placed,
                "message": "Creative DavinciTron canvas generated using signal as a random seed."
            },
            plot_data={"canvas": canvas},
            metadata={"seed": seed, "score": score, "placed": placed},
        )

"""TimeFeatureBlock — ProcessingBlock for time-domain feature extraction."""


from typing import Any, Iterable
import hashlib
import io

from scipy import stats
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

try:
    from .utils import validate_1d_signal
except ImportError:
    from utils import validate_1d_signal


class TimeFeatureBlock(ProcessingBlock):
    """Extract time-domain features from signals and generate visualization."""

    id = "time_features"
    name = "Time-Domain Features"
    category = "feature-extraction"
    input_schema = {"signal": "1d-array", "sample_rate": "positive-float"}
    output_schema = {
        "mean": "float",
        "median": "float",
        "std": "float",
        "variance": "float",
        "rms": "float",
        "peak_to_peak": "float",
        "min": "float",
        "max": "float",
        "skewness": "float",
        "kurtosis": "float",
        "zero_crossing_rate": "float",
        "crest_factor": "float",
        "quality_score": "float",
        "signal_seed": "int",
    }
    default_params = {}

    def validate(self, params: dict[str, Any], sample_rate: float) -> dict[str, Any]:
        """Validate parameters. TimeFeatureBlock has no configurable params."""
        merged = super().validate(params, sample_rate)
        return merged

    def run(
        self, signal: Iterable[float], sample_rate: float, params: dict[str, Any]
    ) -> BlockRunResult:
        """
        Extract time-domain features from signal and generate visualization.

        Args:
            signal: 1D signal array
            sample_rate: Sample rate in Hz
            params: Block parameters (unused for this block)

        Returns:
            BlockRunResult with features dict in result, visualization in plot_data.
        """
        params = self.validate(params, sample_rate)

        # Validate and convert signal
        data = validate_1d_signal(signal, name="signal")
        warnings = []

        # Compute deterministic seed from signal (like DavinciTronBlock)
        signal_bytes = data.tobytes()
        hash_val = hashlib.sha256(signal_bytes).hexdigest()
        signal_seed = int(hash_val, 16) % (2**32)

        # Extract basic statistical features
        features = {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "variance": float(np.var(data)),
            "rms": float(np.sqrt(np.mean(data**2))),
            "peak_to_peak": float(np.max(data) - np.min(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "skewness": float(stats.skew(data)),
            "kurtosis": float(stats.kurtosis(data)),
            "zero_crossing_rate": float(self._zero_crossing_rate(data)),
            "crest_factor": float(self._crest_factor(data)),
        }

        # Compute quality score using multiple criteria (learnable inspection)
        quality_score = self._evaluate_quality(features)

        # Quality checks
        if features["std"] == 0.0:
            warnings.append("Signal is flat (zero variance). Features may not be meaningful.")
        if features["peak_to_peak"] / (features["rms"] + 1e-10) > 10:
            warnings.append("Signal has very high crest factor. May contain clipping or outliers.")
        if np.any(np.abs(data) > 1e6):
            warnings.append("Signal contains very large values. Consider scaling.")

        # Generate visualization as PIL Image
        viz_image = self._visualize_features(data, sample_rate, features)

        # Build result dictionary
        result_dict = dict(features)
        result_dict["quality_score"] = round(quality_score, 4)
        result_dict["signal_seed"] = signal_seed

        return BlockRunResult(
            result=result_dict,
            output_signal=None,  # Features don't modify the signal
            plot_data={"visualization": viz_image},
            warnings=warnings,
            metadata={
                "sample_rate": sample_rate,
                "sample_count": len(data),
                "duration_seconds": len(data) / sample_rate,
                "quality_score": quality_score,
                "signal_seed": signal_seed,
            },
        )

    @staticmethod
    def _zero_crossing_rate(signal: np.ndarray) -> float:
        """
        Compute zero-crossing rate: fraction of samples where sign changes.

        Returns:
            ZCR in [0, 1], where 1 means alternating positive/negative.
        """
        if signal.size < 2:
            return 0.0
        sign_changes = np.abs(np.diff(np.sign(signal)))
        return float(np.mean(sign_changes) / 2.0)

    @staticmethod
    def _crest_factor(signal: np.ndarray) -> float:
        """
        Compute crest factor: peak amplitude / RMS.

        Returns:
            Ratio >= 1. For sine wave, ~1.41. For impulse, can be >> 1.
        """
        rms = np.sqrt(np.mean(signal**2))
        if rms == 0:
            return 0.0
        return float(np.max(np.abs(signal)) / rms)

    @staticmethod
    def _evaluate_quality(features: dict[str, float]) -> float:
        """
        Evaluate signal quality using multiple weighted criteria.
        
        Returns:
            Quality score in [0, 1].
        """
        # Multiple perspective evaluation (inspired by Audience in davincitron)
        scores = []

        # Criterion 1: Variance (higher is generally better for feature extraction)
        var_score = min(features["std"] / (abs(features["mean"]) + 0.1), 1.0)
        scores.append(var_score * 0.3)

        # Criterion 2: Dynamic range utilization
        range_score = features["peak_to_peak"] / (abs(features["mean"]) + 1.0)
        range_score = min(range_score, 1.0)
        scores.append(range_score * 0.25)

        # Criterion 3: Non-uniformity (kurtosis indicates shape complexity)
        kurt_score = min(abs(features["kurtosis"]) / 10.0, 1.0)
        scores.append(kurt_score * 0.2)

        # Criterion 4: Temporal complexity (zero crossing rate)
        zcr_score = features["zero_crossing_rate"]
        scores.append(zcr_score * 0.15)

        # Criterion 5: Moderate crest factor (not too clipped, not too impulsive)
        crest_ideal = 1.41  # sine wave reference
        crest_score = 1.0 - min(abs(features["crest_factor"] - crest_ideal) / 5.0, 1.0)
        scores.append(max(crest_score, 0.0) * 0.1)

        return float(np.mean(scores))

    @staticmethod
    def _visualize_features(
        signal: np.ndarray, sample_rate: float, features: dict[str, float]
    ) -> Image.Image:
        """
        Generate a 2x2 subplot visualization and return as PIL Image.

        Args:
            signal: The signal array
            sample_rate: Sample rate in Hz
            features: Computed feature dictionary

        Returns:
            PIL Image of the visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Time-Domain Feature Analysis", fontsize=14, fontweight='bold')

        # Plot 1: Signal waveform with statistics
        duration = len(signal) / sample_rate
        t = np.linspace(0, duration, len(signal))
        axes[0, 0].plot(t, signal, linewidth=0.8, color='steelblue', label='Signal')
        axes[0, 0].axhline(y=features['mean'], color='red', linestyle='--', 
                          label=f"Mean: {features['mean']:.3f}", linewidth=1)
        axes[0, 0].fill_between(
            t,
            features['mean'] - features['std'],
            features['mean'] + features['std'],
            alpha=0.2, color='red', label=f"±σ: {features['std']:.3f}"
        )
        axes[0, 0].set_xlabel('Time (s)', fontsize=9)
        axes[0, 0].set_ylabel('Amplitude', fontsize=9)
        axes[0, 0].set_title('Waveform with Mean and Std Dev', fontsize=10)
        axes[0, 0].legend(fontsize=8, loc='upper right')
        axes[0, 0].grid(alpha=0.3)

        # Plot 2: Histogram with distribution
        axes[0, 1].hist(signal, bins=40, color='steelblue', alpha=0.7, edgecolor='black', density=True)
        axes[0, 1].axvline(x=features['mean'], color='red', linestyle='--', 
                           label=f"Mean: {features['mean']:.3f}", linewidth=1)
        axes[0, 1].axvline(x=features['median'], color='green', linestyle='--', 
                           label=f"Median: {features['median']:.3f}", linewidth=1)
        axes[0, 1].set_xlabel('Amplitude', fontsize=9)
        axes[0, 1].set_ylabel('Density', fontsize=9)
        axes[0, 1].set_title('Amplitude Distribution', fontsize=10)
        axes[0, 1].legend(fontsize=8, loc='upper right')
        axes[0, 1].grid(alpha=0.3, axis='y')

        # Plot 3: Feature comparison (normalized bar chart)
        feature_names = [
            'rms', 'peak_to_peak', 'std', 'crest_factor',
            'zero_crossing_rate', 'skewness', 'kurtosis'
        ]
        feature_values = np.array([features.get(name, 0.0) for name in feature_names])
        # Normalize for visualization
        feature_values_norm = (feature_values - np.min(feature_values)) / \
                             (np.max(feature_values) - np.min(feature_values) + 1e-10)
        
        colors = plt.cm.viridis(feature_values_norm)
        axes[1, 0].barh(feature_names, feature_values_norm, color=colors)
        axes[1, 0].set_xlabel('Normalized Value', fontsize=9)
        axes[1, 0].set_title('Feature Magnitudes (Normalized)', fontsize=10)
        axes[1, 0].grid(alpha=0.3, axis='x')

        # Plot 4: Summary statistics text box
        summary_text = f"""
SUMMARY STATISTICS

Mean:           {features['mean']:>10.4f}
Median:         {features['median']:>10.4f}
Std Dev:        {features['std']:>10.4f}
Variance:       {features['variance']:>10.4f}

Min:            {features['min']:>10.4f}
Max:            {features['max']:>10.4f}
Peak-to-Peak:   {features['peak_to_peak']:>10.4f}
RMS:            {features['rms']:>10.4f}

Skewness:       {features['skewness']:>10.4f}
Kurtosis:       {features['kurtosis']:>10.4f}
Crest Factor:   {features['crest_factor']:>10.4f}
Zero Cross:     {features['zero_crossing_rate']:>10.4f}
        """
        axes[1, 1].text(0.05, 0.95, summary_text, fontfamily='monospace', fontsize=8,
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Feature Summary', fontsize=10)

        plt.tight_layout()

        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        image = Image.open(buf)
        # Must copy to prevent buffer closure issues
        image = image.copy()
        buf.close()

        return image


def resample_signal(sig: np.ndarray, target_len: int) -> np.ndarray:
    if len(sig) == target_len:
        return sig
    try:
        from scipy.signal import resample
        return resample(sig, target_len)
    except ImportError:
        # fallback simple linear interpolation
        xp = np.linspace(0, 1, len(sig))
        x = np.linspace(0, 1, target_len)
        return np.interp(x, xp, sig)


class SemiclassicalFTBlock(ProcessingBlock):
    id = "sft"
    name = "Semiclassical Fourier Transform"
    category = "frequency"
    input_schema = {"signal": "1d-array", "sample_rate": "positive-float"}
    output_schema = {
        "frequencies": "array",
        "final_sft_proj_probability": "array",
        "final_sft_mod_probability": "array",
        "final_qft_probability": "array",
        "final_dft_probability": "array",
        "mean_fidelity_proj": "float",
        "final_fidelity_proj": "float",
        "mean_fidelity_mod": "float",
        "final_fidelity_mod": "float",
        "steps": "int",
        "alpha": "float",
        "fidelity_proj_over_time": "array",
        "fidelity_mod_over_time": "array",
    }
    default_params = {
        "sft_alpha": 0.5,
        "sft_steps": 50,
        "sft_dt": 0.1,
        "sft_potential": 0.001,
    }

    def validate(self, params: dict[str, Any], sample_rate: float) -> dict[str, Any]:
        merged = super().validate(params, sample_rate)
        try:
            merged["sft_alpha"] = min(1.0, max(0.0, float(merged.get("sft_alpha", 0.5))))
        except (TypeError, ValueError):
            merged["sft_alpha"] = 0.5
        try:
            merged["sft_steps"] = min(500, max(5, int(merged.get("sft_steps", 50))))
        except (TypeError, ValueError):
            merged["sft_steps"] = 50
        try:
            merged["sft_dt"] = min(10.0, max(0.001, float(merged.get("sft_dt", 0.1))))
        except (TypeError, ValueError):
            merged["sft_dt"] = 0.1
        try:
            merged["sft_potential"] = min(1.0, max(0.0, float(merged.get("sft_potential", 0.001))))
        except (TypeError, ValueError):
            merged["sft_potential"] = 0.001
        return merged

    def run(self, signal: Iterable[float], sample_rate: float, params: dict[str, Any]) -> BlockRunResult:
        params = self.validate(params, sample_rate)
        alpha = params["sft_alpha"]
        steps = params["sft_steps"]
        dt = params["sft_dt"]
        kappa = params["sft_potential"]

        sig_in = np.asarray(signal, dtype=float)
        N = 256
        sig = resample_signal(sig_in, N)

        # Normalize to form initial wavefunction psi_0
        norm = np.linalg.norm(sig)
        if norm == 0:
            psi = np.ones(N, dtype=complex) / np.sqrt(N)
        else:
            psi = sig.astype(complex) / norm

        # Embed the user data in complex space as psi_data
        psi_data = psi.copy()

        # Construct Hamiltonian matrix H = K + V
        K = np.zeros((N, N), dtype=complex)
        for i in range(N):
            K[i, i] = 1.0
            K[i, (i + 1) % N] = -0.5
            K[i, (i - 1) % N] = -0.5

        # V (harmonic potential centered at N/2)
        x_indices = np.arange(N)
        V_diag = 0.5 * kappa * (x_indices - N / 2.0) ** 2
        V = np.diag(V_diag).astype(complex)

        H = K + V

        # Crank-Nicolson matrices: (I + i*dt/2 * H) * psi_new = (I - i*dt/2 * H) * psi_old
        I = np.identity(N, dtype=complex)
        M_plus = I + 1j * (dt / 2.0) * H
        M_minus = I - 1j * (dt / 2.0) * H

        # Set up deterministic random generation based on input hash
        import hashlib
        h = hashlib.sha256(sig_in.tobytes()).hexdigest()
        seed = int(h, 16) % (2**32)
        rng = np.random.default_rng(seed)

        # Trace history for plotting
        psi_history = [psi.copy()]
        fidelity_proj_history = []
        fidelity_mod_history = []
        collapsed_history = []
        psi_sc_history = []

        # Precompute the propagator matrix U = M_plus^-1 * M_minus to optimize performance
        # and avoid potential multi-threaded LAPACK solve segfaults on virtualized CPUs.
        try:
            U = np.linalg.inv(M_plus).dot(M_minus)
        except Exception:
            U = None

        for step in range(steps):
            # 1. Evolve under Schrödinger equation
            if U is not None:
                psi = U.dot(psi)
            else:
                psi = np.linalg.solve(M_plus, M_minus.dot(psi))
            psi_norm = np.linalg.norm(psi)
            if psi_norm > 0:
                psi = psi / psi_norm
            psi_history.append(psi.copy())

            # 2. Form probability density and sample collapse position
            P = np.abs(psi) ** 2
            P = P / np.sum(P)  # Normalize to handle numerical precision drift
            collapsed_idx = rng.choice(N, p=P)
            collapsed_history.append(collapsed_idx)

            # 3. Create Dirac delta collapsed state
            psi_col = np.zeros(N, dtype=complex)
            psi_col[collapsed_idx] = 1.0

            # 4. Semiclassical wavefunction interpolation
            # Interpolate alpha_step from 0 to alpha over the steps to simulate continuous collapse
            alpha_step = (step / (steps - 1)) * alpha if steps > 1 else alpha
            psi_sc = (1.0 - alpha_step) * psi + alpha_step * psi_col
            psi_sc_norm = np.linalg.norm(psi_sc)
            if psi_sc_norm > 0:
                psi_sc = psi_sc / psi_sc_norm
            psi_sc_history.append(psi_sc.copy())

            # 5. Semiclassical Operators acting on psi_data
            # Projective Operator: P_sc = |psi_sc><psi_sc|
            # psi_proj = P_sc * psi_data = <psi_sc, psi_data> * psi_sc
            proj_coef = np.vdot(psi_sc, psi_data)
            psi_proj = proj_coef * psi_sc
            psi_proj_norm = np.linalg.norm(psi_proj)
            if psi_proj_norm > 0:
                psi_proj = psi_proj / psi_proj_norm

            # Modulative Operator: M_sc = diag(psi_sc)
            # psi_mod = psi_sc * psi_data (element-wise)
            psi_mod = psi_sc * psi_data
            psi_mod_norm = np.linalg.norm(psi_mod)
            if psi_mod_norm > 0:
                psi_mod = psi_mod / psi_mod_norm

            # 6. Fourier Transforms
            phi = np.fft.fft(psi) / np.sqrt(N)
            P_FT = np.abs(phi) ** 2

            phi_proj = np.fft.fft(psi_proj) / np.sqrt(N)
            P_proj_FT = np.abs(phi_proj) ** 2

            phi_mod = np.fft.fft(psi_mod) / np.sqrt(N)
            P_mod_FT = np.abs(phi_mod) ** 2

            # Calculate spectral fidelities
            fid_proj = float(np.sum(np.sqrt(P_FT * P_proj_FT)) ** 2)
            fid_mod = float(np.sum(np.sqrt(P_FT * P_mod_FT)) ** 2)
            fidelity_proj_history.append(fid_proj)
            fidelity_mod_history.append(fid_mod)

        # Final state calculations
        final_psi = psi_history[-1]
        final_psi_sc = psi_sc_history[-1]

        final_phi = np.fft.fft(final_psi) / np.sqrt(N)
        final_P_FT = np.abs(final_phi) ** 2

        # Final operators acting on user data
        final_proj_coef = np.vdot(final_psi_sc, psi_data)
        final_psi_proj = final_proj_coef * final_psi_sc
        final_psi_proj_norm = np.linalg.norm(final_psi_proj)
        if final_psi_proj_norm > 0:
            final_psi_proj = final_psi_proj / final_psi_proj_norm
        final_phi_proj = np.fft.fft(final_psi_proj) / np.sqrt(N)
        final_P_proj_FT = np.abs(final_phi_proj) ** 2

        final_psi_mod = final_psi_sc * psi_data
        final_psi_mod_norm = np.linalg.norm(final_psi_mod)
        if final_psi_mod_norm > 0:
            final_psi_mod = final_psi_mod / final_psi_mod_norm
        final_phi_mod = np.fft.fft(final_psi_mod) / np.sqrt(N)
        final_P_mod_FT = np.abs(final_phi_mod) ** 2

        # Final classical collapsed state
        final_collapsed_idx = collapsed_history[-1]
        final_psi_col = np.zeros(N, dtype=complex)
        final_psi_col[final_collapsed_idx] = 1.0
        final_phi_col = np.fft.fft(final_psi_col) / np.sqrt(N)
        final_P_col_FT = np.abs(final_phi_col) ** 2

        # Construct Semiclassical Projection Operator Matrix (Outer Product)
        # P_op = |psi_sc><psi_sc|
        P_op = np.outer(final_psi_sc, np.conj(final_psi_sc))

        # Shift frequencies and probabilities for nice plotting
        freqs = np.fft.fftfreq(N, d=1.0 / sample_rate)
        freqs_shifted = np.fft.fftshift(freqs)
        P_FT_shifted = np.fft.fftshift(final_P_FT)
        P_proj_FT_shifted = np.fft.fftshift(final_P_proj_FT)
        P_mod_FT_shifted = np.fft.fftshift(final_P_mod_FT)
        P_col_FT_shifted = np.fft.fftshift(final_P_col_FT)

        result_data = {
            "frequencies": freqs_shifted.tolist(),
            "final_sft_proj_probability": P_proj_FT_shifted.tolist(),
            "final_sft_mod_probability": P_mod_FT_shifted.tolist(),
            "final_qft_probability": P_FT_shifted.tolist(),
            "final_dft_probability": P_col_FT_shifted.tolist(),
            "mean_fidelity_proj": round(float(np.mean(fidelity_proj_history)), 5),
            "final_fidelity_proj": round(fidelity_proj_history[-1], 5),
            "mean_fidelity_mod": round(float(np.mean(fidelity_mod_history)), 5),
            "final_fidelity_mod": round(fidelity_mod_history[-1], 5),
            "steps": steps,
            "alpha": alpha,
            "fidelity_proj_over_time": [round(f, 5) for f in fidelity_proj_history],
            "fidelity_mod_over_time": [round(f, 5) for f in fidelity_mod_history],
        }

        # Keep plot data in original complex/arrays for generator
        plot_data = {
            "N": N,
            "psi_history": [p.tolist() if isinstance(p, np.ndarray) else p for p in psi_history],
            "psi_sc_history": [p.tolist() if isinstance(p, np.ndarray) else p for p in psi_sc_history],
            "collapsed_history": collapsed_history,
            "fidelity_proj_history": fidelity_proj_history,
            "fidelity_mod_history": fidelity_mod_history,
            "freqs": freqs_shifted,
            "final_P_FT": P_FT_shifted,
            "final_P_proj_FT": P_proj_FT_shifted,
            "final_P_mod_FT": P_mod_FT_shifted,
            "final_P_col_FT": P_col_FT_shifted,
            "steps": steps,
            "alpha": alpha,
            "dt": dt,
            "operator_matrix_magnitude": np.abs(P_op).tolist(),
        }

        return BlockRunResult(
            result=result_data,
            output_signal=final_psi_proj.real,
            plot_data=plot_data,
            warnings=[],
            metadata={
                "N": N,
                "final_fidelity_proj": fidelity_proj_history[-1],
                "final_fidelity_mod": fidelity_mod_history[-1],
            }
        )


BLOCK_REGISTRY: dict[str, ProcessingBlock] = {
    block.id: block
    for block in (
        FFTBlock(),
        FilterBlock(),
        WaveletBlock(),
        ifftBlock(),
        DavinciTronBlock(),
        TimeFeatureBlock(),
        SemiclassicalFTBlock(),
    )
}


def get_block(block_id: str) -> ProcessingBlock:
    try:
        return BLOCK_REGISTRY[block_id]
    except KeyError as exc:
        raise ValueError(f"unknown processing block: {block_id}") from exc


def list_blocks() -> list[dict[str, Any]]:
    return [
        {
            "id": block.id,
            "name": block.name,
            "category": block.category,
            "input_schema": block.input_schema,
            "output_schema": block.output_schema,
            "default_params": block.default_params,
        }
        for block in BLOCK_REGISTRY.values()
    ]


def timed_run(block: ProcessingBlock, signal: Iterable[float], sample_rate: float, params: dict[str, Any]) -> tuple[BlockRunResult, float]:
    start = perf_counter()
    run_result = block.run(signal, sample_rate, params)
    elapsed_ms = (perf_counter() - start) * 1000.0
    return run_result, elapsed_ms
