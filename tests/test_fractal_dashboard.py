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
import importlib
import visualization.live_training_dashboard
# Reload to ensure it picks up the mocked tkinter from sys.modules
importlib.reload(visualization.live_training_dashboard)
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
    def test_leaderboard_pnl_formatting(self, mock_subplots):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        dashboard = FractalDashboard(self.root, self.queue)

        # Add mock templates directly
        dashboard.templates = {
            1: {'id': 1, 'pnl': 100, 'count': 10},
            2: {'id': 2, 'pnl': -50, 'count': 5}
        }

        # Trigger update
        dashboard._update_leaderboard()

        # Check insert calls
        # tree_ranks is a mock, so we inspect its insert calls
        insert_calls = dashboard.tree_ranks.insert.call_args_list

        # We expect 2 calls (sorted by PnL desc: 100, then -50)
        self.assertEqual(len(insert_calls), 2)

        # Check first call (Profit)
        args1, kwargs1 = insert_calls[0]
        # args: parent, index
        # insert(parent, index, iid=None, **kw)
        # Check call arguments structure. Mock call arguments are (args, kwargs)
        # insert("", tk.END, values=..., tags=...)

        # Extract kwargs from the call
        # insert call args: ("", tk.END)
        # insert call kwargs: {values: ..., tags: ...}

        call_kwargs1 = insert_calls[0].kwargs
        values1 = call_kwargs1.get('values')
        tags1 = call_kwargs1.get('tags')

        self.assertEqual(values1[0], 1) # ID
        self.assertIn("▲ $100", values1[2]) # PnL string
        self.assertEqual(tags1, ('profit',)) # Tag

        # Check second call (Loss)
        call_kwargs2 = insert_calls[1].kwargs
        values2 = call_kwargs2.get('values')
        tags2 = call_kwargs2.get('tags')

        self.assertEqual(values2[0], 2) # ID
        self.assertIn("▼ $-50", values2[2]) # PnL string
        self.assertEqual(tags2, ('loss',)) # Tag

if __name__ == '__main__':
    # Patching infinite loop in _process_queue for testing
    # We'll just call _handle_message directly or catch the recursion
    # Actually, simpler to just test _handle_message directly
    unittest.main()
