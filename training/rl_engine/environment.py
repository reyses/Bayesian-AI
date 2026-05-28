import numpy as np

class NMPEnvironment:
    """
    NMP (Nightmare Protocol) First Principles Environment
    Calculates Theoretical Exits based on market physics (Variance Ratios, Z-scores).
    """
    
    # Feature Indices corresponding to the 13-feature array
    VELOCITY = 0
    Z_SCORE = 1
    ACCELERATION = 2
    DMI_DIFF = 3
    STD_PRICE = 4
    STD_VOLUME = 5
    VARIANCE_RATIO = 6
    FIB_POSITION = 7
    HTF_Z_SCORE = 8
    SESSION_PHASE = 9
    VOLUME_DELTA = 10
    PRICE_VOL = 11
    DMI_VOL_EXHAUSTION = 12

    @staticmethod
    def check_theoretical_exit(current_state: np.ndarray, is_long: bool) -> bool:
        """
        Determines if the structural boundaries demand an exit.
        """
        z_score = current_state[NMPEnvironment.Z_SCORE]
        vr = current_state[NMPEnvironment.VARIANCE_RATIO]
        exhaustion = current_state[NMPEnvironment.DMI_VOL_EXHAUSTION]

        # 1. Stable Regime (Mean Reverting)
        if vr < 1.0:
            if is_long and z_score >= 0.0:
                return True
            if not is_long and z_score <= 0.0:
                return True

        # 2. Chaotic Regime (Trending)
        if vr > 1.0:
            # Exits only on violent Roche Limit breach
            if is_long and z_score > 3.0:
                return True
            if not is_long and z_score < -3.0:
                return True

        # 3. Structural Exhaustion
        if exhaustion > 0.9:
            return True

        return False

    @staticmethod
    def calculate_regret(agent_pnl: float, theoretical_nmp_pnl: float) -> float:
        """
        Calculates the Hindsight Experience Replay regret.
        R_regret = -| MFE_theoretical - PnL_actual |
        Penalizes the agent only if it left money on the table compared to pure math.
        """
        if theoretical_nmp_pnl > agent_pnl:
            return -abs(theoretical_nmp_pnl - agent_pnl)
        return 0.0
