"""
Shared engine factory — creates TBN + ExecutionEngine from a CheckpointBundle.

Used by both trainer.py and live_engine.py to avoid duplicated initialization.
"""
import logging
from typing import Optional, Dict, Set

from core.checkpoint_loader import CheckpointBundle
from core.timeframe_belief_network import TimeframeBeliefNetwork
from core.execution_engine import ExecutionEngine
from core.exit_engine import ExitEngine
from core.fractal_clustering import FractalClusteringEngine

logger = logging.getLogger(__name__)


def create_belief_network(bundle: CheckpointBundle,
                          engine) -> TimeframeBeliefNetwork:
    """Create TBN from checkpoint bundle.

    Args:
        bundle: Loaded checkpoint data.
        engine: StatisticalFieldEngine instance.
    """
    tbn = TimeframeBeliefNetwork(
        pattern_library=bundle.pattern_library,
        scaler=bundle.scaler,
        engine=engine,
        valid_tids=bundle.valid_tids,
        centroids_scaled=bundle.centroids_scaled,
    )
    logger.info(f"  Belief network: {len(TimeframeBeliefNetwork.TIMEFRAMES_SECONDS)} TF workers "
                f"(conviction threshold: {TimeframeBeliefNetwork.MIN_CONVICTION:.2f})")
    return tbn


def create_execution_engine(
    bundle: CheckpointBundle,
    brain,
    belief_network: TimeframeBeliefNetwork,
    exit_engine: ExitEngine,
    tick_size: float,
    point_value: float,
    mode: str = 'is',
    tier_preference: bool = False,
    bias_threshold: float = 0.55,
    dmi_threshold: float = 0.0,
    depth_only: Optional[int] = None,
    looseness: int = 0,
) -> ExecutionEngine:
    """Create ExecutionEngine from checkpoint bundle + mode-specific params.

    Args:
        bundle: Loaded checkpoint data.
        brain: MarketBayesianBrain instance.
        belief_network: TBN instance.
        exit_engine: ExitEngine instance.
        tick_size: Asset tick size (e.g. 0.25).
        point_value: Asset point value (e.g. 2.0).
        mode: 'is', 'oos', or 'live'.
        tier_preference: Enable tier-based score adjustment for tiebreaker.
        bias_threshold: Gate threshold for direction bias.
        dmi_threshold: Gate threshold for DMI.
        depth_only: If set, restrict to single depth (for depth analysis).
        looseness: Gate looseness level (0=default, 1-4=progressively looser).
    """
    tier_score_adj = ({1: -1.5, 2: -0.5, 3: 0.0, 4: 0.5}
                      if tier_preference or mode == 'live' else {})

    ee = ExecutionEngine(
        brain=brain,
        belief_network=belief_network,
        exit_engine=exit_engine,
        pattern_library=bundle.pattern_library,
        scaler=bundle.scaler,
        centroids_scaled=bundle.centroids_scaled,
        valid_tids=bundle.valid_tids,
        tick_size=tick_size,
        point_value=point_value,
        mode=mode,
        tier_score_adj=tier_score_adj,
        depth_score_adj=bundle.depth_score_adj,
        template_tier_map=bundle.template_tier_map,
        bias_threshold=bias_threshold,
        dmi_threshold=dmi_threshold,
        depth_filter_out=bundle.depth_filter_out,
        depth_only=depth_only,
        feature_extractor=FractalClusteringEngine().extract_features,
        looseness=looseness,
    )
    logger.info(f"  Execution engine: mode={mode}" +
                (f" looseness={looseness}" if looseness else ""))
    return ee
