import sys
import os
import torch
import numpy as np
from dataclasses import dataclass

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.fractal_clustering import FractalClusteringEngine

@dataclass
class DummyPattern:
    z_score: float
    velocity: float
    momentum: float
    coherence: float
    timeframe: str
    depth: int
    parent_type: str

def main():
    print("Verifying FractalClusteringEngine with CUDAKMeans...")

    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is NOT available. This test will run on CPU (sklearn).")

    # Generate dummy data
    n_samples = 600 # > 500 to trigger MiniBatchKMeans if CPU, but we want to test CUDA path
    manifest = []

    rng = np.random.RandomState(42)

    print(f"Generating {n_samples} dummy patterns...")
    for i in range(n_samples):
        p = DummyPattern(
            z_score=rng.randn(),
            velocity=rng.randn(),
            momentum=rng.randn(),
            coherence=rng.rand(),
            timeframe='15s',
            depth=0,
            parent_type='ROCHE_SNAP' if rng.rand() > 0.5 else ''
        )
        manifest.append(p)

    engine = FractalClusteringEngine(n_clusters=10, max_variance=0.5)

    print("Running create_templates()...")
    templates = engine.create_templates(manifest)

    print(f"Created {len(templates)} templates.")
    if len(templates) > 0:
        print("First template centroid:", templates[0].centroid)
        print("First template member count:", templates[0].member_count)

    # Check if any template has members
    total_members = sum(t.member_count for t in templates)
    print(f"Total members in templates: {total_members} (should be {n_samples})")

    assert total_members == n_samples, "Total members mismatch!"

    # Test refine_clusters (fission)
    if len(templates) > 0:
        tmpl = templates[0]
        # Generate dummy params
        member_params = []
        for _ in range(tmpl.member_count):
            member_params.append({
                'stop_loss_ticks': rng.randint(10, 50),
                'take_profit_ticks': rng.randint(20, 100),
                'trailing_stop_ticks': rng.randint(5, 20)
            })

        print(f"Testing refine_clusters on Template {tmpl.template_id}...")
        new_tmpls = engine.refine_clusters(tmpl.template_id, member_params, tmpl.patterns)
        print(f"Refine result: {len(new_tmpls)} new templates (0 means no split)")

    print("\nVerification COMPLETE.")

if __name__ == "__main__":
    main()
