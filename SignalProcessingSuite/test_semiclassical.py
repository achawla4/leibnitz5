"""Unit tests and Crank-Nicolson physical simulation analysis for SemiclassicalFTBlock."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from blocks import SemiclassicalFTBlock, BLOCK_REGISTRY
except ImportError:
    from SignalProcessingSuite.blocks import SemiclassicalFTBlock, BLOCK_REGISTRY


class TestSemiclassicalFTBlock:
    """Test suite for the Semiclassical Fourier Transform block with operator evolution."""

    @pytest.fixture
    def block(self):
        """Provide a fresh SemiclassicalFTBlock instance."""
        return SemiclassicalFTBlock()

    def test_block_metadata(self, block):
        """Verify block is correctly registered with proper metadata."""
        assert block.id == "sft"
        assert block.name == "Semiclassical Fourier Transform"
        assert block.category == "frequency"
        assert "sft_alpha" in block.default_params
        assert "sft_steps" in block.default_params

    def test_constant_signal(self, block):
        """Test SFT on constant signal: verifying keys and boundary alpha cases."""
        signal = np.ones(256) * 3.0
        sample_rate = 1000.0

        # When alpha = 0.0, the semiclassical wavefunction is EXACTLY the Schrödinger evolved wave.
        # Thus, the SFT spectral fidelity is exactly 1.0 at all steps.
        result = block.run(signal, sample_rate, {"sft_alpha": 0.0, "sft_steps": 20})
        res = result.result

        assert res["alpha"] == 0.0
        assert res["steps"] == 20
        assert res["final_fidelity_proj"] == pytest.approx(1.0, abs=1e-5)
        assert res["mean_fidelity_proj"] == pytest.approx(1.0, abs=1e-5)

    def test_wave_interpolation_boundaries(self, block):
        """Verify boundaries: alpha=0.0 is perfect quantum fidelity, alpha=1.0 is classical Dirac limit."""
        fs = 500.0
        t = np.linspace(0, 1, 256, endpoint=False)
        signal = np.sin(2 * np.pi * 10 * t)

        # Alpha = 0.0: Perfect fidelity for projection
        res_quantum = block.run(signal, fs, {"sft_alpha": 0.0, "sft_steps": 30}).result
        assert res_quantum["final_fidelity_proj"] == pytest.approx(1.0, abs=1e-5)
        assert res_quantum["final_fidelity_mod"] > 0.2

        # Alpha = 1.0: Fully collapsed Dirac state
        res_classical = block.run(signal, fs, {"sft_alpha": 1.0, "sft_steps": 30}).result
        assert res_classical["final_fidelity_proj"] < 0.45  # Highly degraded due to collapse

    def test_multi_body_particle_types(self, block):
        """Verify 2-body simulation works for Distinguishable, Boson, and Fermion particle types."""
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
        fs = 500.0

        for ptype in ["distinguishable", "bosons", "fermions"]:
            res = block.run(signal, fs, {
                "sft_particle_type": ptype,
                "sft_grid_size": 16,  # small size for fast tests
                "sft_steps": 10
            }).result
            assert "final_fidelity_proj" in res
            assert res["final_fidelity_proj"] > 0.0

    def test_multi_body_interaction_potential(self, block):
        """Verify varying the 2-body interaction coupling (attractive vs repulsive)."""
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
        fs = 500.0

        # Run with attractive coupling (-0.5) and repulsive coupling (1.0)
        res_attr = block.run(signal, fs, {"sft_interaction": -0.5, "sft_grid_size": 16, "sft_steps": 10}).result
        res_rep = block.run(signal, fs, {"sft_interaction": 1.0, "sft_grid_size": 16, "sft_steps": 10}).result

        assert res_attr["final_fidelity_proj"] > 0.0
        assert res_rep["final_fidelity_proj"] > 0.0



def run_tradeoff_analysis():
    """Run SFT across a range of alpha values to analyze the quantum-to-classical interpolation."""
    print("=" * 95)
    print("Semiclassical Fourier Transform (SFT) Projective & Modulative Semiclassical Operators Analysis")
    print("=" * 95)
    
    # Generate a two-tone signal: 10 Hz and 35 Hz
    fs = 500.0
    t = np.linspace(0, 1.0, 256, endpoint=False)
    signal = 1.0 * np.sin(2 * np.pi * 10.0 * t) + 0.5 * np.sin(2 * np.pi * 35.0 * t)
    
    block = SemiclassicalFTBlock()
    
    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    
    print(f"Input Signal: 10 Hz + 35 Hz Multitone (N = 256, Fs = {fs} Hz, Steps = 50, dt = 0.1)")
    print("-" * 95)
    print(f"{'Alpha (a)':<10} | {'Mean Fid (Proj)':<17} | {'Final Fid (Proj)':<18} | {'Mean Fid (Mod)':<17} | {'Final Fid (Mod)':<17}")
    print("-" * 95)
    
    for a in alphas:
        # Run SFT
        res_obj = block.run(signal, fs, {
            "sft_alpha": a,
            "sft_steps": 50,
            "sft_dt": 0.1,
            "sft_potential": 0.001
        })
        res = res_obj.result
        print(f"{res['alpha']:<10.3f} | {res['mean_fidelity_proj']:<17.5f} | {res['final_fidelity_proj']:<18.5f} | "
              f"{res['mean_fidelity_mod']:<17.5f} | {res['final_fidelity_mod']:<17.5f}")

    print("-" * 95)
    print("Evolution Step Test: alpha = 0.5, dt = 0.1, varying Steps")
    print("-" * 95)
    for steps in [10, 50, 100, 200]:
        res_obj = block.run(signal, fs, {
            "sft_alpha": 0.5,
            "sft_steps": steps,
            "sft_dt": 0.1,
            "sft_potential": 0.001
        })
        res = res_obj.result
        print(f"{res['steps']:<10} | {res['mean_fidelity_proj']:<17.5f} | {res['final_fidelity_proj']:<18.5f} | "
              f"{res['mean_fidelity_mod']:<17.5f} | {res['final_fidelity_mod']:<17.5f}")
        
    print("=" * 95)
    print("Analysis complete.")


if __name__ == "__main__":
    run_tradeoff_analysis()
