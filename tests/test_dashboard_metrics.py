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
mock_tk.Label = MagicMock()
mock_tk.Text = MagicMock()
mock_tk.Button = MagicMock()

mock_ttk.Frame = MagicMock()
mock_ttk.Label = MagicMock()
mock_ttk.Button = MagicMock()
mock_ttk.Style = MagicMock()
mock_ttk.Separator = MagicMock()
mock_ttk.Treeview = MagicMock()

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

class TestDashboardMetrics(unittest.TestCase):
    def setUp(self):
        self.root = MagicMock()

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_metrics_update(self, mock_thread):
        # Prevent thread from starting
        mock_thread.return_value.start = MagicMock()

        dashboard = LiveDashboard(self.root)

        # Override the label tuples with mocks to track config calls
        # LiveDashboard creates tuples like self._card_wr = (label_val, label_sub)
        dashboard._card_wr = (MagicMock(), MagicMock())
        dashboard._card_pnl = (MagicMock(), MagicMock())

        # Sample Data
        dashboard.data = {
            'iteration': 10,
            'total_iterations': 100,
            'elapsed_seconds': 3600,
            'states_learned': 50,
            'high_confidence_states': 10,
            'trades': [
                {'result': 'WIN', 'pnl': 100.0},
                {'result': 'LOSS', 'pnl': -50.0},
                {'result': 'WIN', 'pnl': 20.0}
            ],
            # Calculated fields usually come from backend but refresh_dashboard calculates some display logic
            'cumulative_win_rate': 66.7,
            'total_pnl': 70.0,
            'today_pnl': 70.0,
            'today_win_rate': 0.667
        }

        # Call refresh
        dashboard.refresh_dashboard()

        # Check Win Rate Label (index 0 of tuple)
        # Expected: "66.7%" and Green
        lbl_wr_val = dashboard._card_wr[0]
        call_args = lbl_wr_val.config.call_args
        self.assertIsNotNone(call_args, "Win Rate label config not called")
        self.assertEqual(call_args[1].get('text'), "66.7%")
        self.assertEqual(call_args[1].get('fg'), "#00ff00")

        # Check P&L Label (index 0 of tuple)
        # Expected: "$70.00" and Green
        lbl_pnl_val = dashboard._card_pnl[0]
        call_args = lbl_pnl_val.config.call_args
        self.assertIsNotNone(call_args, "P&L label config not called")
        self.assertEqual(call_args[1].get('text'), "$70.00")
        self.assertEqual(call_args[1].get('fg'), "#00ff00")

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_metrics_update_negative(self, mock_thread):
        # Test negative scenarios (Losses)
        mock_thread.return_value.start = MagicMock()
        dashboard = LiveDashboard(self.root)

        # Override
        dashboard._card_wr = (MagicMock(), MagicMock())
        dashboard._card_pnl = (MagicMock(), MagicMock())

        dashboard.data = {
            'iteration': 10,
            'trades': [
                {'result': 'LOSS', 'pnl': -100.0},
                {'result': 'LOSS', 'pnl': -50.0}
            ],
            'cumulative_win_rate': 0.0,
            'total_pnl': -150.0,
            'today_pnl': -150.0,
            'today_win_rate': 0.0
        }

        dashboard.refresh_dashboard()

        # Win Rate: 0% -> Red
        lbl_wr_val = dashboard._card_wr[0]
        call_args = lbl_wr_val.config.call_args
        self.assertEqual(call_args[1].get('text'), "0.0%")
        self.assertEqual(call_args[1].get('fg'), "#ff4444")

        # P&L: -150.0 -> Red
        lbl_pnl_val = dashboard._card_pnl[0]
        call_args = lbl_pnl_val.config.call_args
        self.assertEqual(call_args[1].get('text'), "$-150.00")
        self.assertEqual(call_args[1].get('fg'), "#ff4444")

if __name__ == '__main__':
    unittest.main()
