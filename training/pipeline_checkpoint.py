"""
Pipeline Checkpoint Manager
Handles persistence across all three phases of the fractal training pipeline:
  Phase 2:   Discovery manifest (PatternEvents across all TF levels)
  Phase 2.5: Clustering templates (PatternTemplates)
  Phase 3:   Optimization scheduler state (completed/pending templates)

Checkpoint files:
  pipeline_state.json     — Human-readable phase tracker
  discovery_manifest.pkl  — Phase 2 output
  discovery_levels.json   — Which TF levels completed (for partial resume)
  templates.pkl           — Phase 2.5 output
  scheduler_state.pkl     — Phase 3 progress (completed + pending queue)
"""
import os
import json
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


class PipelineCheckpoint:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # File paths
        self.state_path = os.path.join(checkpoint_dir, 'pipeline_state.json')
        self.manifest_path = os.path.join(checkpoint_dir, 'discovery_manifest.pkl')
        self.levels_path = os.path.join(checkpoint_dir, 'discovery_levels.json')
        self.templates_path = os.path.join(checkpoint_dir, 'templates.pkl')
        self.scheduler_path = os.path.join(checkpoint_dir, 'scheduler_state.pkl')

    # ------------------------------------------------------------------
    # Pipeline state (human-readable JSON)
    # ------------------------------------------------------------------

    def update_phase(self, phase: str, status: str, metadata: Dict = None):
        """Update current pipeline phase in JSON tracker."""
        state = self._load_json(self.state_path) or {}
        state['current_phase'] = phase
        state['status'] = status
        state['updated_at'] = datetime.now().isoformat()
        if metadata:
            state.update(metadata)
        self._save_json(self.state_path, state)

    def get_phase(self) -> Optional[Dict]:
        """Read current pipeline state."""
        return self._load_json(self.state_path)

    # ------------------------------------------------------------------
    # Phase 2: Discovery manifest
    # ------------------------------------------------------------------

    def save_discovery(self, manifest: List[Any], completed_levels: List[str]):
        """Save discovery results and which TF levels are done."""
        self._save_pickle(self.manifest_path, manifest)
        self._save_json(self.levels_path, {
            'completed_levels': completed_levels,
            'pattern_count': len(manifest),
            'saved_at': datetime.now().isoformat()
        })
        self.update_phase('discovery', 'complete', {
            'discovery_patterns': len(manifest),
            'discovery_levels': len(completed_levels)
        })
        print(f"  [CHECKPOINT] Discovery saved: {len(manifest)} patterns, {len(completed_levels)} levels")

    def save_discovery_level(self, manifest_so_far: List[Any], completed_levels: List[str]):
        """Save after each level completes (incremental)."""
        self._save_pickle(self.manifest_path, manifest_so_far)
        self._save_json(self.levels_path, {
            'completed_levels': completed_levels,
            'pattern_count': len(manifest_so_far),
            'saved_at': datetime.now().isoformat()
        })

    def load_discovery(self) -> Tuple[Optional[List], List[str]]:
        """Load discovery manifest and completed levels."""
        manifest = self._load_pickle(self.manifest_path)
        levels_data = self._load_json(self.levels_path)
        completed = levels_data.get('completed_levels', []) if levels_data else []
        return manifest, completed

    def has_discovery(self) -> bool:
        return os.path.exists(self.manifest_path) and os.path.exists(self.levels_path)

    # ------------------------------------------------------------------
    # Phase 2.5: Clustering templates
    # ------------------------------------------------------------------

    def save_templates(self, templates: List[Any]):
        """Save clustered templates."""
        self._save_pickle(self.templates_path, templates)
        self.update_phase('clustering', 'complete', {
            'template_count': len(templates)
        })
        print(f"  [CHECKPOINT] Templates saved: {len(templates)} templates")

    def load_templates(self) -> Optional[List]:
        return self._load_pickle(self.templates_path)

    def has_templates(self) -> bool:
        return os.path.exists(self.templates_path)

    # ------------------------------------------------------------------
    # Phase 3: Scheduler state (optimization progress)
    # ------------------------------------------------------------------

    def save_scheduler_state(self, completed_results: Dict, fissioned_ids: set,
                              pending_queue: List[Any], metrics: Dict = None):
        """
        Save Phase 3 optimization progress.
        completed_results: {template_id: result_dict} for DONE templates
        fissioned_ids: set of template_ids that were SPLIT
        pending_queue: remaining PatternTemplate objects to process
        """
        state = {
            'completed_results': completed_results,
            'fissioned_ids': fissioned_ids,
            'pending_queue': pending_queue,
            'metrics': metrics or {},
            'saved_at': time.time()
        }
        self._save_pickle(self.scheduler_path, state)

        n_done = len(completed_results)
        n_fissioned = len(fissioned_ids)
        n_pending = len(pending_queue)
        self.update_phase('optimization', 'in_progress', {
            'optimized': n_done,
            'fissioned': n_fissioned,
            'pending': n_pending
        })

    def load_scheduler_state(self) -> Tuple[Optional[Dict], Optional[set], Optional[List], Optional[Dict]]:
        """Load Phase 3 state. Returns (completed, fissioned_ids, pending_queue, metrics) or Nones."""
        state = self._load_pickle(self.scheduler_path)
        if state is None:
            return None, None, None, None
        return (
            state.get('completed_results', {}),
            state.get('fissioned_ids', set()),
            state.get('pending_queue', []),
            state.get('metrics', {})
        )

    def has_scheduler_state(self) -> bool:
        return os.path.exists(self.scheduler_path)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def clear(self):
        """Wipe all checkpoint files for a fresh start."""
        for path in [self.state_path, self.manifest_path, self.levels_path,
                     self.templates_path, self.scheduler_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"  [CHECKPOINT] Removed: {os.path.basename(path)}")
        print("  [CHECKPOINT] All pipeline checkpoints cleared.")

    def summary(self) -> str:
        """Print checkpoint status summary."""
        lines = ["Pipeline Checkpoint Status:"]
        state = self.get_phase()
        if state:
            lines.append(f"  Phase: {state.get('current_phase', '?')} ({state.get('status', '?')})")
            lines.append(f"  Updated: {state.get('updated_at', '?')}")
        else:
            lines.append("  No checkpoint found (fresh run)")

        if self.has_discovery():
            _, levels = self.load_discovery()
            lines.append(f"  Discovery: {len(levels)} levels completed")
        if self.has_templates():
            lines.append(f"  Templates: checkpoint exists")
        if self.has_scheduler_state():
            completed, fissioned, pending, _ = self.load_scheduler_state()
            lines.append(f"  Scheduler: {len(completed)} done, {len(fissioned)} fissioned, {len(pending)} pending")

        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_pickle(self, path: str, data: Any):
        tmp_path = path + '.tmp'
        with open(tmp_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)

    def _load_pickle(self, path: str) -> Optional[Any]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  WARNING: Failed to load checkpoint {os.path.basename(path)}: {e}")
            return None

    def _save_json(self, path: str, data: Dict):
        tmp_path = path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, path)

    def _load_json(self, path: str) -> Optional[Dict]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  WARNING: Failed to load {os.path.basename(path)}: {e}")
            return None
