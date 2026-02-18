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

        # Add templates
        t1 = {'id': 1, 'pnl': 100, 'count': 10}
        t2 = {'id': 2, 'pnl': 200, 'count': 5}
        t3 = {'id': 3, 'pnl': 50, 'count': 20}
        dashboard.templates = {1: t1, 2: t2, 3: t3}

        # 1. Default Sort (PnL Descending): 2 ($200), 1 ($100), 3 ($50)
        dashboard._update_leaderboard()
        # Get calls to insert
        calls = dashboard.tree_ranks.insert.call_args_list
        # We expect 3 calls.
        self.assertEqual(len(calls), 3)
        ids = [call.kwargs['values'][0] for call in calls]
        self.assertEqual(ids, [2, 1, 3])

        # 2. Click "ID" (New col -> default Descending): 3, 2, 1
        dashboard.tree_ranks.insert.reset_mock()
        dashboard._on_header_click("ID")
        calls = dashboard.tree_ranks.insert.call_args_list
        ids = [call.kwargs['values'][0] for call in calls]
        self.assertEqual(ids, [3, 2, 1])

        # 3. Click "ID" again (Same col -> toggle to Ascending): 1, 2, 3
        dashboard.tree_ranks.insert.reset_mock()
        dashboard._on_header_click("ID")
        calls = dashboard.tree_ranks.insert.call_args_list
        ids = [call.kwargs['values'][0] for call in calls]
        self.assertEqual(ids, [1, 2, 3])

        # 4. Click "Count" (New col -> default Descending): 3 (20), 1 (10), 2 (5)
        dashboard.tree_ranks.insert.reset_mock()
        dashboard._on_header_click("Count")
        calls = dashboard.tree_ranks.insert.call_args_list
        ids = [call.kwargs['values'][0] for call in calls]
        self.assertEqual(ids, [3, 1, 2])

if __name__ == '__main__':
    # Patching infinite loop in _process_queue for testing
    # We'll just call _handle_message directly or catch the recursion
    # Actually, simpler to just test _handle_message directly
    unittest.main()
