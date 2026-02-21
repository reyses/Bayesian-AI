import sys
import os
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.append(os.getcwd())

from training.fractal_dna_tree import FractalDNATree
from training.fractal_discovery_agent import PatternEvent
from core.three_body_state import ThreeBodyQuantumState

def verify_dna_tree():
    print("Verifying FractalDNATree...")
    try:
        tree = FractalDNATree(n_clusters_per_level=2)

        # Create dummy patterns
        patterns = []
        for i in range(20):
            state = ThreeBodyQuantumState.null_state()
            # Hack to set read-only attributes or assume default works if we can't change it easily
            # Since we can't change frozen state easily, we rely on the fact that null_state has defaults.
            # But clustering needs variation.
            # We can use object.__setattr__ to modify frozen dataclass for testing
            object.__setattr__(state, 'z_score', float(i % 5))
            object.__setattr__(state, 'rel_volume', float(1.0 + i*0.1))

            p = PatternEvent(
                pattern_type='ROCHE_SNAP',
                timestamp=1234567890.0 + i,
                price=100.0,
                z_score=float(i % 5),
                velocity=0.1,
                momentum=5.0,
                coherence=0.8,
                file_source='test',
                idx=i,
                state=state,
                timeframe='15s',
                oracle_marker=1 if i % 2 == 0 else -1
            )
            # Add dummy parent chain
            p.parent_chain = [
                {'tf': '1h', 'z': 1.0, 'velocity': 0.1, 'mom': 1.0, 'adx': 20.0, 'hurst': 0.6, 'dmi_plus': 10.0, 'dmi_minus': 5.0, 'pid': 0.0, 'osc_coh': 0.0, 'rel_volume': 1.0},
                {'tf': '15m', 'z': 2.0, 'velocity': 0.2, 'mom': 2.0, 'adx': 25.0, 'hurst': 0.7, 'dmi_plus': 15.0, 'dmi_minus': 5.0, 'pid': 0.0, 'osc_coh': 0.0, 'rel_volume': 1.2},
                {'tf': '5m', 'z': 3.0, 'velocity': 0.3, 'mom': 3.0, 'adx': 30.0, 'hurst': 0.8, 'dmi_plus': 20.0, 'dmi_minus': 5.0, 'pid': 0.0, 'osc_coh': 0.0, 'rel_volume': 1.5},
            ]
            patterns.append(p)

        tree.fit(patterns)
        print("Tree fitted.")

        if tree.root is None:
            print("FAIL: Root is None")
            return False

        # Match
        dna, node, conf = tree.match(patterns[0])
        print(f"Match result: DNA={dna}, Node={node.node_id if node else 'None'}, Conf={conf}")

        if dna is None:
            print("FAIL: Match returned None")
            return False

        print(f"DNA Key: {dna.key}")
        if dna.key.startswith('L|') or dna.key.startswith('S|'):
            print("FAIL: DNA key still has direction prefix")
            return False

        print("PASS: FractalDNATree verification successful")
        return True

    except Exception as e:
        print(f"FAIL: Error verifying tree: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if verify_dna_tree():
        print("ALL TESTS PASSED")
    else:
        print("TESTS FAILED")
        sys.exit(1)
