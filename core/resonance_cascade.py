"""
Resonance Cascade Detector
Identifies when all 9 timeframes align → Harmonic amplification
"""
import numpy as np
from dataclasses import dataclass
from typing import List
from core.three_body_state import ThreeBodyQuantumState

@dataclass
class ResonanceState:
    """Cross-timeframe phase alignment metrics"""
    phase_coherence: float       # 0-1, alignment strength
    resonance_frequency: float
    amplitude_multiplier: float  # 1-4+ energy amplification
    layer_phases: List[float]
    layer_alignment: List[bool]
    resonance_type: str         # NONE|PARTIAL|FULL|CRITICAL
    cascade_probability: float  # P(flash move in 60s)
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    volume_damping: float
    volatility_damping: float
    news_catalyst: bool
    liquidity_vacuum: bool
    timestamp: float
    time_to_cascade: float

class ResonanceCascadeDetector:
    """Detects harmonic alignment across 9 timeframes"""
    
    def __init__(self):
        self.PARTIAL_RESONANCE = 0.60
        self.FULL_RESONANCE = 0.80
        self.CRITICAL_RESONANCE = 0.95
    
    def detect_resonance(
        self,
        quantum_state: ThreeBodyQuantumState,
        layer_deviations: dict,  # {L1: z_score, ...}
        layer_velocities: dict,
        volume_profile: dict,
        order_book_depth: float,
        news_events: List[str]
    ) -> ResonanceState:
        """
        Detect harmonic resonance building
        When all layers synchronize → cascade imminent
        """
        layer_phases = self._calculate_layer_phases(layer_deviations, layer_velocities)
        phase_coherence, alignment_vector = self._measure_phase_coherence(layer_phases)
        amplitude_mult = (1.0 + phase_coherence) ** 2
        
        energies = {
            'kinetic': sum(v**2 for v in layer_velocities.values()) / 2.0,
            'potential': sum(d**2 for d in layer_deviations.values()) / 2.0
        }
        energies['total'] = energies['kinetic'] + energies['potential']
        
        damping = {
            'volume': min(volume_profile.get('current_volume', 1000) / 
                         (volume_profile.get('avg_volume', 1000) + 1), 2.0),
            'volatility': min(order_book_depth / 10000.0, 1.0)
        }
        
        if phase_coherence < self.PARTIAL_RESONANCE:
            resonance_type = 'NONE'
        elif phase_coherence < self.FULL_RESONANCE:
            resonance_type = 'PARTIAL'
        elif phase_coherence < self.CRITICAL_RESONANCE:
            resonance_type = 'FULL'
        else:
            resonance_type = 'CRITICAL' if energies['total'] > 5.0 else 'FULL'
        
        # Cascade probability
        base_probs = {'NONE': 0.01, 'PARTIAL': 0.10, 'FULL': 0.40, 'CRITICAL': 0.85}
        cascade_prob = base_probs[resonance_type]
        cascade_prob *= min(energies['total'] / 10.0, 2.0)
        cascade_prob /= ((damping['volume'] + damping['volatility']) / 2.0 + 0.1)
        if len(news_events) > 0:
            cascade_prob *= 2.0
        cascade_prob = min(cascade_prob, 0.98)
        
        time_to_cascade = 60.0 / (cascade_prob * energies['total']) if cascade_prob > 0.50 else 999.0
        
        return ResonanceState(
            phase_coherence=phase_coherence,
            resonance_frequency=0.0,
            amplitude_multiplier=amplitude_mult,
            layer_phases=layer_phases,
            layer_alignment=alignment_vector,
            resonance_type=resonance_type,
            cascade_probability=cascade_prob,
            kinetic_energy=energies['kinetic'],
            potential_energy=energies['potential'],
            total_energy=energies['total'],
            volume_damping=damping['volume'],
            volatility_damping=damping['volatility'],
            news_catalyst=len(news_events) > 0,
            liquidity_vacuum=order_book_depth < 1000,
            timestamp=quantum_state.timestamp,
            time_to_cascade=time_to_cascade
        )
    
    def _calculate_layer_phases(self, deviations, velocities):
        """Phase θ = arctan2(velocity, displacement)"""
        phases = []
        for layer in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9']:
            phase = np.arctan2(velocities.get(layer, 0.0), deviations.get(layer, 0.0))
            if phase < 0:
                phase += 2*np.pi
            phases.append(phase)
        return phases
    
    def _measure_phase_coherence(self, phases):
        """Order parameter R = |Σexp(iθ)| / N"""
        N = len(phases)
        order_param = sum(np.exp(1j * theta) for theta in phases) / N
        coherence = abs(order_param)
        mean_phase = np.angle(order_param)
        alignment_vector = [
            abs(theta - mean_phase) < np.pi/4 or abs(theta - mean_phase) > 7*np.pi/4
            for theta in phases
        ]
        return coherence, alignment_vector
