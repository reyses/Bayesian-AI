import unittest
import sys
import os
import shutil
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

def create_mock_label(*args, **kwargs):
    return MagicMock()

mock_tk.Label = MagicMock(side_effect=create_mock_label)
mock_tk.Text = MagicMock()
mock_tk.Button = MagicMock()

mock_ttk.Frame = MagicMock()
mock_ttk.Label = MagicMock(side_effect=create_mock_label)
mock_ttk.Button = MagicMock()
mock_ttk.Style = MagicMock()
mock_ttk.Separator = MagicMock()

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
from visualization.live_training_dashboard import LiveDashboard
import importlib
import visualization.live_training_dashboard

class TestDashboardMetrics(unittest.TestCase):
    def setUp(self):
        # Reload module to avoid cross-test contamination
        sys.modules['tkinter'] = mock_tk
        sys.modules['tkinter.ttk'] = mock_ttk
        importlib.reload(visualization.live_training_dashboard)

        self.root = MagicMock()

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_metrics_update(self, mock_thread):
        # Prevent thread from starting
        mock_thread.return_value.start = MagicMock()

        dashboard = LiveDashboard(self.root)

        # Sample Data
        dashboard.data = {
            'iteration': 10,
            'total_iterations': 100,
            'elapsed_seconds': 3600,
            'states_learned': 50,
            'high_confidence_states': 10,
            'cumulative_win_rate': 66.666,
            'total_pnl': 70.0,
            'trades': [
                {'result': 'WIN', 'pnl': 100.0},
                {'result': 'LOSS', 'pnl': -50.0},
                {'result': 'WIN', 'pnl': 20.0}
            ]
        }

        # Call refresh
        dashboard.refresh_dashboard()

        # Verify Win Rate Logic
        # Wins: 2, Total: 3 => 66.66...% -> 66.7%
        # Expected color: Green (#00ff00) because >= 50%
        # Note: We check if the calls contain the expected arguments.
        # Since config might be called with different args, we check the specific call.

        # Using any_order=True or checking specific call args
        # dashboard.lbl_wr.config.assert_called_with(text="Win Rate: 66.7%", foreground="#00ff00")
        # However, the code might call config multiple times or with different kwargs.
        # Let's check the call args directly.

        # Check Win Rate
        # dashboard._card_wr is (value_label, sub_label)
        call_args_wr = dashboard._card_wr[0].config.call_args
        self.assertIsNotNone(call_args_wr, "lbl_wr.config was not called")
        # Check positional args if text is positional, or kwargs if named
        # config(text=...) usually
        self.assertEqual(call_args_wr[1].get('text'), "66.7%")
        self.assertEqual(call_args_wr[1].get('fg'), "#00ff00")

        # Check P&L
        # Total: 70.0
        call_args_pnl = dashboard._card_pnl[0].config.call_args
        self.assertIsNotNone(call_args_pnl, "lbl_pnl.config was not called")
        self.assertEqual(call_args_pnl[1].get('text'), "$70.00")
        self.assertEqual(call_args_pnl[1].get('fg'), "#00ff00")

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_metrics_update_negative(self, mock_thread):
        # Test negative scenarios (Losses)
        mock_thread.return_value.start = MagicMock()
        dashboard = LiveDashboard(self.root)

        dashboard.data = {
            'iteration': 10,
            'cumulative_win_rate': 0.0,
            'total_pnl': -150.0,
            'trades': [
                {'result': 'LOSS', 'pnl': -100.0},
                {'result': 'LOSS', 'pnl': -50.0}
            ]
        }

        dashboard.refresh_dashboard()

        # Win Rate: 0% -> Red
        call_args_wr = dashboard._card_wr[0].config.call_args
        self.assertEqual(call_args_wr[1].get('text'), "0.0%")
        self.assertEqual(call_args_wr[1].get('fg'), "#ff4444")

        # P&L: -150.0 -> Red
        call_args_pnl = dashboard._card_pnl[0].config.call_args
        self.assertEqual(call_args_pnl[1].get('text'), "$-150.00")
        self.assertEqual(call_args_pnl[1].get('fg'), "#ff4444")

if __name__ == '__main__':
    unittest.main()
