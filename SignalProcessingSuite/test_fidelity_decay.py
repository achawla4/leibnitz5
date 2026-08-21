"""Script to verify that spectral fidelity decays as the semiclassical wavefunction collapses."""

from __future__ import annotations

import sys
import os
import numpy as np

# Ensure SignalProcessingSuite is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from blocks import SemiclassicalFTBlock
except ImportError:
    from SignalProcessingSuite.blocks import SemiclassicalFTBlock


def verify_fidelity_decay():
    print("=" * 80)
    print("Semiclassical Fourier Transform: Step-by-Step Spectral Fidelity Analysis")
    print("=" * 80)

    # Generate a simple 12 Hz sinusoidal signal
    fs = 1000.0
    t = np.linspace(0, 1.0, 256, endpoint=False)
    signal = np.sin(2 * np.pi * 12.0 * t)

    block = SemiclassicalFTBlock()
    steps = 50
    alpha = 0.8  # Strong collapse to highlight the difference

    res_obj = block.run(signal, fs, {
        "sft_alpha": alpha,
        "sft_steps": steps,
        "sft_dt": 0.1,
        "sft_potential": 0.001
    })
    res = res_obj.result

    proj_fidelities = res["fidelity_proj_over_time"]
    mod_fidelities = res["fidelity_mod_over_time"]

    print(f"Total steps: {steps}")
    print(f"Target Alpha (final collapse): {alpha}")
    print("-" * 80)
    print(f"{'Step':<10} | {'Alpha_t':<15} | {'Projective Fidelity':<25} | {'Modulative Fidelity':<25}")
    print("-" * 80)

    for i in range(steps):
        alpha_t = (i / (steps - 1)) * alpha if steps > 1 else alpha
        # Print every 5 steps and the last step
        if i % 5 == 0 or i == steps - 1:
            print(f"{i:<10} | {alpha_t:<15.4f} | {proj_fidelities[i]:<25.6f} | {mod_fidelities[i]:<25.6f}")

    print("-" * 80)

    # Assertions to verify the requested physics/behavior:
    # 1. Initial projective fidelity (step 0, alpha=0) must be exactly 1.0
    assert abs(proj_fidelities[0] - 1.0) < 1e-5, f"Initial projective fidelity was {proj_fidelities[0]} instead of 1.0!"
    
    # 2. Later projective fidelity must be lesser than the initial one
    assert proj_fidelities[-1] < proj_fidelities[0], f"Fidelity did not decrease! Final: {proj_fidelities[-1]} vs Initial: {proj_fidelities[0]}"
    
    print("SUCCESS: SFT Projective Fidelity successfully starts at 1.0 (perfect quantum match)")
    print(f"and decays to {proj_fidelities[-1]:.6f} (lesser match due to classical Dirac delta collapse).")
    print("=" * 80)


if __name__ == "__main__":
    verify_fidelity_decay()
