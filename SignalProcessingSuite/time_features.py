# -*- coding: utf-8 -*-
"""
Time-Domain Feature Extraction System

time_features.py — A hierarchical signal analyzer

The TimeFeatureTron is a meta-analyzer inspired by the idea of nested measurements.
It coordinates multiple worker extractors (each computing local signal properties)
and a higher-level inspector that validates and synthesizes these measurements.
Each layer has its own validation rules and perception of signal quality.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Iterable

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


# ===============================================================
# SignalValidator — structural feasibility and sanity check
# ===============================================================

class SignalValidator:
    """Ensures signals are valid and appropriately scaled."""

    def __init__(self, max_value=1e6, min_samples=2):
        self.max_value = max_value
        self.min_samples = min_samples

    def validate_1d_signal(self, signal: Iterable[float], name: str = "signal") -> np.ndarray:
        """Convert and validate 1D signal array."""
        data = np.asarray(signal, dtype=np.float64).flatten()
        
        if data.size < self.min_samples:
            raise ValueError(f"{name} must have at least {self.min_samples} samples, got {data.size}")
        
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError(f"{name} contains NaN or Inf values")
        
        return data

    def approve(self, data: np.ndarray) -> tuple[bool, list[str]]:
        """Check if signal is suitable for analysis."""
        warnings = []
        
        if np.std(data) == 0.0:
            warnings.append("Signal is flat (zero variance). Features may not be meaningful.")
        
        if np.max(np.abs(data)) > self.max_value:
            warnings.append(f"Signal contains values > {self.max_value}. Consider scaling.")
        
        return True, warnings


# ===============================================================
# FeatureExtractor — computational measurement engine
# ===============================================================

class FeatureExtractor:
    """Computes statistical and temporal features from signals."""

    def __init__(self, validator: SignalValidator):
        self.validator = validator

    def extract_basic_stats(self, data: np.ndarray) -> dict[str, float]:
        """Extract fundamental statistical measures."""
        return {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "variance": float(np.var(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
        }

    def extract_power_metrics(self, data: np.ndarray) -> dict[str, float]:
        """Extract amplitude and energy-related features."""
        rms = float(np.sqrt(np.mean(data**2)))
        peak_to_peak = float(np.max(data) - np.min(data))
        crest_factor = self._crest_factor(data)
        
        return {
            "rms": rms,
            "peak_to_peak": peak_to_peak,
            "crest_factor": crest_factor,
        }

    def extract_shape_metrics(self, data: np.ndarray) -> dict[str, float]:
        """Extract distribution shape features."""
        return {
            "skewness": float(stats.skew(data)),
            "kurtosis": float(stats.kurtosis(data)),
        }

    def extract_temporal_metrics(self, data: np.ndarray) -> dict[str, float]:
        """Extract time-domain and transition features."""
        return {
            "zero_crossing_rate": float(self._zero_crossing_rate(data)),
        }

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


# ===============================================================
# SignalVisualizer — generates visual representations
# ===============================================================

class SignalVisualizer:
    """Creates PNG visualizations of signals and their features."""

    @staticmethod
    def visualize_signal_and_features(
        signal: np.ndarray,
        sample_rate: float,
        features: dict[str, float],
        output_path: str,
    ) -> None:
        """Create a comprehensive visualization of signal and extracted features."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Signal Analysis Report", fontsize=14, fontweight='bold')

        # Plot 1: Signal waveform
        duration = len(signal) / sample_rate
        t = np.linspace(0, duration, len(signal))
        axes[0, 0].plot(t, signal, linewidth=0.8, color='steelblue')
        axes[0, 0].axhline(y=features['mean'], color='red', linestyle='--', label=f"Mean: {features['mean']:.3f}")
        axes[0, 0].axhline(y=features['mean'] + features['std'], color='orange', linestyle=':', alpha=0.7, label=f"±σ: {features['std']:.3f}")
        axes[0, 0].axhline(y=features['mean'] - features['std'], color='orange', linestyle=':', alpha=0.7)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Waveform')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(alpha=0.3)

        # Plot 2: Histogram/distribution
        axes[0, 1].hist(signal, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=features['mean'], color='red', linestyle='--', label=f"Mean: {features['mean']:.3f}")
        axes[0, 1].axvline(x=features['median'], color='green', linestyle='--', label=f"Median: {features['median']:.3f}")
        axes[0, 1].set_xlabel('Amplitude')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Distribution')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(alpha=0.3, axis='y')

        # Plot 3: Feature values (bar chart)
        feature_names = list(features.keys())[:8]  # Top 8 for readability
        feature_values = [features[name] for name in feature_names]
        # Normalize for visualization
        feature_values_norm = np.array(feature_values)
        feature_values_norm = (feature_values_norm - np.min(feature_values_norm)) / (np.max(feature_values_norm) - np.min(feature_values_norm) + 1e-10)
        
        colors = plt.cm.viridis(feature_values_norm)
        axes[1, 0].barh(feature_names, feature_values_norm, color=colors)
        axes[1, 0].set_xlabel('Normalized Value')
        axes[1, 0].set_title('Feature Magnitudes')
        axes[1, 0].grid(alpha=0.3, axis='x')

        # Plot 4: Quality metrics text box
        metrics_text = f"""
        RMS: {features['rms']:.4f}
        Peak-to-Peak: {features['peak_to_peak']:.4f}
        Crest Factor: {features['crest_factor']:.4f}
        Zero Cross Rate: {features['zero_crossing_rate']:.4f}
        Skewness: {features['skewness']:.4f}
        Kurtosis: {features['kurtosis']:.4f}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontfamily='monospace', fontsize=9,
                        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Key Metrics')

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)


# ===============================================================
# QualityInspector — validates and synthesizes measurements
# ===============================================================

class QualityInspector:
    """Aggregates multiple measurements and assesses overall signal quality."""

    def __init__(self, size=3, seed=None):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.criteria = [self._make_criterion() for _ in range(size)]

    def _make_criterion(self):
        """Each inspector has a slightly different sensitivity to features."""
        weights = self.rng.random(12)  # 12 output features
        return lambda feats: float(np.dot(weights, list(feats.values())))

    def evaluate(self, features: dict[str, float]) -> float:
        """Compute overall quality score based on multiple criteria."""
        scores = [c(features) for c in self.criteria]
        return float(np.mean(scores))


# ===============================================================
# WorkerAnalyzer — a subtask analyzer (extracts features)
# ===============================================================

class WorkerAnalyzer:
    """Specialized analyzer that extracts all feature categories from a signal."""

    def __init__(self, validator: SignalValidator, extractor: FeatureExtractor, inspector: QualityInspector):
        self.validator = validator
        self.extractor = extractor
        self.inspector = inspector

    def analyze_signal(self, signal: Iterable[float], sample_rate: float) -> dict[str, Any]:
        """Perform complete feature extraction and quality assessment."""
        # Validate
        data = self.validator.validate_1d_signal(signal, name="signal")
        is_valid, warnings = self.validator.approve(data)

        # Extract features from multiple perspectives
        features = {}
        features.update(self.extractor.extract_basic_stats(data))
        features.update(self.extractor.extract_power_metrics(data))
        features.update(self.extractor.extract_shape_metrics(data))
        features.update(self.extractor.extract_temporal_metrics(data))

        # Evaluate quality
        quality_score = self.inspector.evaluate(features)

        return {
            "features": features,
            "quality_score": quality_score,
            "warnings": warnings,
            "metadata": {
                "sample_rate": sample_rate,
                "sample_count": len(data),
                "duration_seconds": len(data) / sample_rate,
            },
        }


# ===============================================================
# MasterAnalyzer — orchestrates workers and synthesizes results
# ===============================================================

class MasterAnalyzer:
    """High-level analyzer that coordinates feature extraction across multiple signals."""

    def __init__(self, n_workers: int = 3, seed: int = None):
        self.validator = SignalValidator()
        self.extractor = FeatureExtractor(self.validator)
        self.workers = []
        self.master_inspector = QualityInspector(size=5, seed=seed)

        if seed is not None:
            np.random.seed(seed)

        for i in range(n_workers):
            inspector_seed = None if seed is None else seed + i + 10
            inspector = QualityInspector(seed=inspector_seed)
            worker = WorkerAnalyzer(self.validator, self.extractor, inspector)
            self.workers.append(worker)

    def analyze_batch(self, signals: list[np.ndarray], sample_rates: list[float]) -> dict[str, Any]:
        """Analyze multiple signals and synthesize results."""
        results = []
        overall_score = 0.0

        for idx, (signal, sr) in enumerate(zip(signals, sample_rates)):
            worker = self.workers[idx % len(self.workers)]
            result = worker.analyze_signal(signal, sr)
            results.append(result)
            overall_score += result["quality_score"]

        overall_score /= len(results) if results else 1.0

        return {
            "results": results,
            "overall_score": overall_score,
            "signal_count": len(results),
        }


# ===============================================================
# TimeFeatureTron — orchestrates the complete analysis system
# ===============================================================

class TimeFeatureTron:
    """Meta-analyzer that coordinates feature extraction and visualization."""

    def __init__(self, outdir: str = "./time_features_outputs"):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.master = MasterAnalyzer(n_workers=3, seed=42)
        self.visualizer = SignalVisualizer()
        self.run_log = []

    def run(self, test_signals: int = 3):
        """Generate synthetic test signals, analyze them, and visualize."""
        for i in range(test_signals):
            # Generate synthetic test signal with variation
            duration = 2.0  # seconds
            sample_rate = 1000.0  # Hz
            t = np.linspace(0, duration, int(sample_rate * duration))

            # Create signal with multiple components (varies per iteration)
            freq1 = 10 + i * 5  # 10, 15, 20 Hz
            freq2 = 25 + i * 3  # 25, 28, 31 Hz
            signal = (
                np.sin(2 * np.pi * freq1 * t) +           # Primary component
                0.5 * np.sin(2 * np.pi * freq2 * t) +     # Secondary component
                0.1 * np.random.randn(len(t))             # Noise
            )

            # Analyze
            result = self.master.analyze_batch([signal], [sample_rate])
            feature_result = result["results"][0]

            # Generate visualization and save as PNG
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            png_filename = f"time_features_signal_{i:02d}_score{feature_result['quality_score']:.3f}_{ts}.png"
            png_path = os.path.join(self.outdir, png_filename)

            self.visualizer.visualize_signal_and_features(
                signal,
                sample_rate,
                feature_result["features"],
                png_path
            )

            # Log with davincitron-style format
            self.run_log.append({
                "path": png_path,
                "score": feature_result["quality_score"],
                "features": len(feature_result["features"]),
            })

            print(f"Saved {png_path} score={feature_result['quality_score']:.3f} features={len(feature_result['features'])}")
            time.sleep(1)

        # Save run log (matching davincitron format)
        log_path = os.path.join(self.outdir, "time_features_run_log.json")
        with open(log_path, "w") as f:
            json.dump(self.run_log, f, indent=2)
        
        print(f"Run log saved to {log_path}")


# ===============================================================
# MAIN
# ===============================================================

if __name__ == "__main__":
    t = TimeFeatureTron()
    t.run(test_signals=3)
    print("TimeFeatureTron analysis complete.")
