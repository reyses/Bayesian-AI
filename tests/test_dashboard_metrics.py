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

        # Inject mock labels to capture config calls
        # The dashboard uses tuples for cards: (value_label, sub_label)
        mock_wr_val = MagicMock()
        mock_wr_sub = MagicMock()
        dashboard._card_wr = (mock_wr_val, mock_wr_sub)

        mock_pnl_val = MagicMock()
        mock_pnl_sub = MagicMock()
        dashboard._card_pnl = (mock_pnl_val, mock_pnl_sub)

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
            ]
        }
        # Precompute values that LiveDashboard calculates from trades
        # The dashboard logic actually computes these from 'cumulative_win_rate' and 'total_pnl' in data
        # So we must provide those in dashboard.data OR ensure LiveDashboard calculates them.
        # Looking at LiveDashboard.refresh_dashboard:
        # total_pnl = d.get('total_pnl', 0.0)
        # cum_wr = d.get('cumulative_win_rate', 0.0)

        # So we need to provide these pre-calculated values in data for the test to pass
        dashboard.data['total_pnl'] = 70.0
        dashboard.data['cumulative_win_rate'] = 66.7

        # Call refresh
        dashboard.refresh_dashboard()

        # Verify Win Rate Logic
        # Wins: 2, Total: 3 => 66.66...% -> 66.7%
        # Expected color: Green (#00ff00) because >= 50%

        # Check Win Rate
        call_args_wr = mock_wr_val.config.call_args
        self.assertIsNotNone(call_args_wr, "lbl_wr.config was not called")
        self.assertEqual(call_args_wr[1].get('text'), "66.7%")
        self.assertEqual(call_args_wr[1].get('fg'), "#00ff00")

        # Check P&L
        # Total: 70.0
        call_args_pnl = mock_pnl_val.config.call_args
        self.assertIsNotNone(call_args_pnl, "lbl_pnl.config was not called")
        self.assertEqual(call_args_pnl[1].get('text'), "$70.00")
        self.assertEqual(call_args_pnl[1].get('fg'), "#00ff00")

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_metrics_update_negative(self, mock_thread):
        # Test negative scenarios (Losses)
        mock_thread.return_value.start = MagicMock()
        dashboard = LiveDashboard(self.root)

        # Inject mocks
        mock_wr_val = MagicMock()
        mock_wr_sub = MagicMock()
        dashboard._card_wr = (mock_wr_val, mock_wr_sub)

        mock_pnl_val = MagicMock()
        mock_pnl_sub = MagicMock()
        dashboard._card_pnl = (mock_pnl_val, mock_pnl_sub)

        dashboard.data = {
            'iteration': 10,
            'total_pnl': -150.0,
            'cumulative_win_rate': 0.0,
            'trades': [
                {'result': 'LOSS', 'pnl': -100.0},
                {'result': 'LOSS', 'pnl': -50.0}
            ]
        }

        dashboard.refresh_dashboard()

        # Win Rate: 0% -> Red
        call_args_wr = mock_wr_val.config.call_args
        self.assertEqual(call_args_wr[1].get('text'), "0.0%")
        self.assertEqual(call_args_wr[1].get('fg'), "#ff4444")

        # P&L: -150.0 -> Red
        call_args_pnl = mock_pnl_val.config.call_args
        self.assertEqual(call_args_pnl[1].get('text'), "$-150.00")
        self.assertEqual(call_args_pnl[1].get('fg'), "#ff4444")

if __name__ == '__main__':
    unittest.main()
