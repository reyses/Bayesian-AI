import unittest
import sys
import os
import queue
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

# --- CRITICAL: Mock TclError as an Exception ---
class MockTclError(Exception):
    pass
mock_tk.TclError = MockTclError

# --- CRITICAL: Ensure Widgets return unique mocks ---
mock_tk.Text.side_effect = lambda *a, **k: MagicMock()
mock_tk.Menu.side_effect = lambda *a, **k: MagicMock()
mock_ttk.Label.side_effect = lambda *a, **k: MagicMock()
mock_ttk.Frame.side_effect = lambda *a, **k: MagicMock()
mock_ttk.Treeview.side_effect = lambda *a, **k: MagicMock()

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

class TestDashboardUX(unittest.TestCase):
    def setUp(self):
        self.root = MagicMock()
        self.queue = MockQueue()

    @patch('visualization.live_training_dashboard.plt.subplots')
    def test_log_context_menu_actions(self, mock_subplots):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        dashboard = FractalDashboard(self.root, self.queue)

        # Test Clear Log
        dashboard._log_clear()
        dashboard.log_text.delete.assert_called_with("1.0", mock_tk.END)

        # Test Copy All
        dashboard.log_text.get.return_value = "Log Content"
        dashboard._log_copy_all()
        # Verify get call
        dashboard.log_text.get.assert_called_with("1.0", mock_tk.END)
        # Verify clipboard actions
        self.root.clipboard_clear.assert_called()
        self.root.clipboard_append.assert_called_with("Log Content")

    @patch('visualization.live_training_dashboard.plt.subplots')
    def test_log_copy_selection_success(self, mock_subplots):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        dashboard = FractalDashboard(self.root, self.queue)

        # Mock selection retrieval success
        dashboard.log_text.get.return_value = "Selected Text"
        dashboard._log_copy_sel()

        # Verify get call for selection
        dashboard.log_text.get.assert_called_with(mock_tk.SEL_FIRST, mock_tk.SEL_LAST)
        # Verify clipboard actions
        self.root.clipboard_clear.assert_called()
        self.root.clipboard_append.assert_called_with("Selected Text")

    @patch('visualization.live_training_dashboard.plt.subplots')
    def test_log_copy_selection_failure(self, mock_subplots):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        dashboard = FractalDashboard(self.root, self.queue)

        # Mock selection retrieval failure (TclError)
        dashboard.log_text.get.side_effect = mock_tk.TclError("No selection")

        # Should not raise exception
        try:
            dashboard._log_copy_sel()
        except Exception as e:
            self.fail(f"_log_copy_sel raised exception: {e}")

        # Verify clipboard actions were NOT called (since it failed)
        # Note: In _log_copy_sel, clear/append happen AFTER get()
        # So if get() raises, they shouldn't run.
        # But wait, MagicMock calls are cumulative. We should reset mocks or check carefully.
        self.root.clipboard_clear.reset_mock()
        self.root.clipboard_append.reset_mock()

        # Call again to be sure
        dashboard._log_copy_sel()
        self.root.clipboard_clear.assert_not_called()

if __name__ == '__main__':
    unittest.main()
