import unittest
import sys
import os
import queue
import threading
import time
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- MOCK TKINTER & MATPLOTLIB ---
mock_tk = MagicMock()
mock_ttk = MagicMock()
mock_mpl = MagicMock()
mock_plt = MagicMock()
mock_tkagg = MagicMock()
mock_figure = MagicMock()

sys.modules['tkinter'] = mock_tk
sys.modules['tkinter.ttk'] = mock_ttk
sys.modules['matplotlib'] = mock_mpl
sys.modules['matplotlib.pyplot'] = mock_plt
sys.modules['matplotlib.backends'] = MagicMock()
sys.modules['matplotlib.backends.backend_tkagg'] = mock_tkagg
sys.modules['matplotlib.figure'] = mock_figure

# Link ttk to tk
mock_tk.ttk = mock_ttk

# Mock Queue
class MockQueue:
    def __init__(self):
        self.messages = []

    def put(self, item):
        self.messages.append(item)

    def get_nowait(self):
        if not self.messages:
            raise queue.Empty
        return self.messages.pop(0)

# Import Dashboard
from visualization.live_training_dashboard import FractalDashboard

class TestFractalDashboard(unittest.TestCase):
    def setUp(self):
        self.root = MagicMock()
        self.queue = MockQueue()

    @patch('visualization.live_training_dashboard.plt.subplots')
    def test_dashboard_initialization(self, mock_subplots):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        dashboard = FractalDashboard(self.root, self.queue)

        # Verify initial state
        self.assertEqual(dashboard.templates, {})
        self.assertEqual(dashboard.fission_events, [])

    @patch('visualization.live_training_dashboard.plt.subplots')
    def test_template_update(self, mock_subplots):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        dashboard = FractalDashboard(self.root, self.queue)

        # Send Template Update
        msg = {'type': 'TEMPLATE_UPDATE', 'id': 101, 'z': 2.1, 'mom': 4.5, 'pnl': 120, 'count': 45}
        self.queue.put(msg)

        # Process Queue
        try:
            dashboard._process_queue()
        except Exception:
            pass # Stop infinite loop

        # Verify state update
        self.assertIn(101, dashboard.templates)
        self.assertEqual(dashboard.templates[101]['pnl'], 120)

        # Verify Plot Update
        mock_ax.scatter.assert_called()

    @patch('visualization.live_training_dashboard.plt.subplots')
    def test_fission_event(self, mock_subplots):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        dashboard = FractalDashboard(self.root, self.queue)

        # Send Fission Event
        msg = {'type': 'FISSION_EVENT', 'parent_id': 101, 'children_count': 3, 'reason': 'Variance limit'}
        self.queue.put(msg)

        # Process Queue
        try:
            dashboard._process_queue()
        except Exception:
            pass

        # Verify state update
        self.assertEqual(len(dashboard.fission_events), 1)
        self.assertEqual(dashboard.fission_events[0]['parent_id'], 101)

    @patch('visualization.live_training_dashboard.plt.subplots')
    def test_leaderboard_sorting(self, mock_subplots):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        dashboard = FractalDashboard(self.root, self.queue)

        # Setup dummy data
        dashboard.templates = {
            1: {'id': 1, 'pnl': 100, 'win_rate': 0.6, 'count': 10},
            2: {'id': 2, 'pnl': 200, 'win_rate': 0.5, 'count': 20},
            3: {'id': 3, 'pnl': 50, 'win_rate': 0.7, 'count': 5}
        }

        # Mock _update_leaderboard to verify calls if needed, but we can check internal state or just call it
        # Since _on_header_click calls _update_leaderboard, let's verify the sorting logic inside _update_leaderboard
        # But _update_leaderboard updates the Treeview. We can't easily inspect the Treeview because it's mocked.
        # However, we can verify that _sort_col and _sort_reverse are updated correctly.

        # Initial state: PnL Descending
        self.assertEqual(dashboard._sort_col, "PnL")
        self.assertTrue(dashboard._sort_reverse)

        # Click "Win%" -> Win% Descending
        dashboard._on_header_click("Win%")
        self.assertEqual(dashboard._sort_col, "Win%")
        self.assertTrue(dashboard._sort_reverse)

        # Click "Win%" again -> Win% Ascending
        dashboard._on_header_click("Win%")
        self.assertEqual(dashboard._sort_col, "Win%")
        self.assertFalse(dashboard._sort_reverse)

        # Click "Trades" -> Trades Descending
        dashboard._on_header_click("Trades")
        self.assertEqual(dashboard._sort_col, "Trades")
        self.assertTrue(dashboard._sort_reverse)

if __name__ == '__main__':
    # Patching infinite loop in _process_queue for testing
    # We'll just call _handle_message directly or catch the recursion
    # Actually, simpler to just test _handle_message directly
    unittest.main()
