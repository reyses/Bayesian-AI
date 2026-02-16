import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create Mocks
mock_tk = MagicMock()
mock_ttk = MagicMock()
mock_messagebox = MagicMock()
mock_mpl = MagicMock()
mock_plt = MagicMock()
mock_tkagg = MagicMock()
mock_figure = MagicMock()
mock_dates = MagicMock()
mock_style = MagicMock()

# Configure specific mocks
mock_tk.Tk = MagicMock()
mock_tk.Toplevel = MagicMock()
mock_tk.Label = MagicMock()
mock_tk.Text = MagicMock()
mock_tk.Button = MagicMock()

mock_ttk.Frame = MagicMock()
mock_ttk.Label = MagicMock()
mock_ttk.Button = MagicMock()
mock_ttk.Style = MagicMock()
mock_ttk.Separator = MagicMock()
mock_ttk.Treeview = MagicMock()
mock_ttk.Scrollbar = MagicMock()

# Link mock_tk.ttk to mock_ttk
mock_tk.ttk = mock_ttk

# Assign to sys.modules
sys.modules['tkinter'] = mock_tk
sys.modules['tkinter.ttk'] = mock_ttk
sys.modules['tkinter.messagebox'] = mock_messagebox
sys.modules['matplotlib'] = mock_mpl
sys.modules['matplotlib.pyplot'] = mock_plt
sys.modules['matplotlib.backends'] = MagicMock()
sys.modules['matplotlib.backends.backend_tkagg'] = mock_tkagg
sys.modules['matplotlib.figure'] = mock_figure
sys.modules['matplotlib.dates'] = mock_dates
sys.modules['matplotlib.style'] = mock_style

# Now import the dashboard
from visualization.live_training_dashboard import LiveDashboard, Tooltip

class TestDashboardUX(unittest.TestCase):
    def setUp(self):
        self.root = MagicMock()

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_tooltips_presence(self, mock_thread):
        mock_thread.return_value.start = MagicMock()
        dashboard = LiveDashboard(self.root)

        widgets_to_check = [
            dashboard._card_progress[0],
            dashboard._card_pnl[0],
            dashboard._card_wr[0],
            dashboard._card_sharpe[0],
            dashboard.btn_pause,
            dashboard.btn_stop
        ]

        for w in widgets_to_check:
            bind_calls = w.bind.call_args_list
            bound_events = [c[0][0] for c in bind_calls]
            self.assertIn('<Enter>', bound_events, f"Widget {w} missing <Enter> binding for Tooltip")
            self.assertIn('<Leave>', bound_events, f"Widget {w} missing <Leave> binding for Tooltip")

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_status_icons(self, mock_thread):
        mock_thread.return_value.start = MagicMock()
        dashboard = LiveDashboard(self.root)

        status_tests = {
            "RUNNING": "🟢",
            "PAUSED": "⏸️",
            "STOPPED": "🛑",
        }

        for status, icon in status_tests.items():
            dashboard.remote_status = status
            dashboard.lbl_status.config.reset_mock()

            dashboard.update_gui()

            call_args = dashboard.lbl_status.config.call_args
            self.assertIsNotNone(call_args, f"Status update failed for {status}")

            text_arg = call_args[1].get('text', '')
            self.assertIn(icon, text_arg)
            self.assertIn(status, text_arg)

    @patch('visualization.live_training_dashboard.ttk')
    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_log_scrollbar(self, mock_thread, mock_ttk_local):
        """
        Verify that Scrollbar is instantiated and connected.
        We patch 'ttk' in the module namespace to catch the call robustly.
        """
        mock_thread.return_value.start = MagicMock()

        # Configure the mock to behave like a class
        # mock_ttk_local.Scrollbar() should return a mock instance

        dashboard = LiveDashboard(self.root)

        # Check if Scrollbar was instantiated
        self.assertTrue(mock_ttk_local.Scrollbar.called, "Scrollbar not instantiated")

        # Verify call args
        call_args = mock_ttk_local.Scrollbar.call_args
        # Should contain command=dashboard.txt_log.yview

        # txt_log is created using tk.Text (which is mocked globally as mock_tk.Text)
        # So dashboard.txt_log is a mock.

        # Check command
        if call_args:
             kwargs = call_args.kwargs
             self.assertEqual(kwargs.get('command'), dashboard.txt_log.yview)

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_shortcuts_binding(self, mock_thread):
        mock_thread.return_value.start = MagicMock()
        dashboard = LiveDashboard(self.root)

        bind_calls = self.root.bind.call_args_list
        bound_keys = [call[0][0] for call in bind_calls]

        expected_keys = ['<space>', 'p', 'P', 's', 'S', 'x', 'X']

        for key in expected_keys:
            self.assertIn(key, bound_keys, f"Key {key} should be bound")

if __name__ == '__main__':
    unittest.main()
