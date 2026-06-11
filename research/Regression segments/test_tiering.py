import numpy as np
from tiering import classify_tier, max_consecutive

def test_max_consecutive():
    assert max_consecutive(np.array([False, True, True, False])) == 2
    assert max_consecutive(np.array([True, True, True])) == 3
    assert max_consecutive(np.array([False, False])) == 0

def test_tiering_stage2():
    E = 1.0
    # Perfect fit
    assert classify_tier(np.zeros(10), E, max_tier=8) == 1
    
    # Boundary exactly at threshold
    # t=1: max_res <= 1.5*E, out_lo > 1.0*E
    res = np.zeros(10)
    res[0] = 1.5
    assert classify_tier(res, E, max_tier=8) == 1
    res[0] = 1.51
    assert classify_tier(res, E, max_tier=8) == 2

    # Consecutive outliers
    res = np.zeros(10)
    res[0:2] = 1.1 # 2 outliers > 1.0*E
    assert classify_tier(res, E, max_tier=8) == 1 # < 3 passes

    res[0:3] = 1.1 # 3 outliers > 1.0*E -> fails t=1, checks t=2
    # For t=2, max=1.1 (<=2.0), lo=1.5. 1.1 is NOT > 1.5, so consecutive=0 < 3. Passes t=2.
    assert classify_tier(res, E, max_tier=8) == 2

    # Huge residuals
    res = np.ones(10) * 10.0
    assert classify_tier(res, E, max_tier=8) == 9

def test_tiering_stage1_equivalence():
    # old stage1 categorize_segment
    def categorize_segment_old(Y_clean, preds, E):
        residuals = np.abs(Y_clean - preds)
        max_res = np.max(residuals)
        if max_res <= 1.5 * E:
            out_10 = residuals > 1.0 * E
            if max_consecutive(out_10) < 3:
                return 1
        if max_res <= 2.0 * E:
            out_15 = residuals > 1.5 * E
            if max_consecutive(out_15) < 3:
                return 2
        return 3

    for _ in range(100):
        E = np.random.uniform(0.1, 5.0)
        res = np.random.uniform(0.0, 3.0 * E, size=50)
        Y_clean = res
        preds = np.zeros(50)
        
        old_tier = categorize_segment_old(Y_clean, preds, E)
        new_tier = classify_tier(res, E, max_tier=2)
        assert old_tier == new_tier
