import unittest
import sys
import os
import shutil
from unittest.mock import MagicMock, patch
import importlib

class TestDashboardMetrics(unittest.TestCase):
    def setUp(self):
        # Create fresh mocks
        self.mock_tk = MagicMock()
        self.mock_ttk = MagicMock()
        self.mock_messagebox = MagicMock()
        self.mock_mpl = MagicMock()
        self.mock_plt = MagicMock()
        self.mock_tkagg = MagicMock()
        self.mock_figure = MagicMock()
        self.mock_dates = MagicMock()
        self.mock_style = MagicMock()

        # Configure specific mocks
        self.mock_tk.Tk = MagicMock()
        self.mock_tk.Toplevel = MagicMock()
        self.mock_tk.Label = MagicMock()
        self.mock_tk.Text = MagicMock()
        self.mock_tk.Button = MagicMock()

        self.mock_ttk.Frame = MagicMock()
        self.mock_ttk.Label = MagicMock()
        self.mock_ttk.Button = MagicMock()
        self.mock_ttk.Style = MagicMock()
        self.mock_ttk.Separator = MagicMock()

        # Link mock_tk.messagebox
        self.mock_tk.messagebox = self.mock_messagebox

        # Patch sys.modules
        self.patcher = patch.dict(sys.modules, {
            'tkinter': self.mock_tk,
            'tkinter.ttk': self.mock_ttk,
            'tkinter.messagebox': self.mock_messagebox,
            'matplotlib': self.mock_mpl,
            'matplotlib.pyplot': self.mock_plt,
            'matplotlib.backends': MagicMock(),
            'matplotlib.backends.backend_tkagg': self.mock_tkagg,
            'matplotlib.figure': self.mock_figure,
            'matplotlib.dates': self.mock_dates,
            'matplotlib.style': self.mock_style,
        })
        self.patcher.start()

        # Reload the dashboard module
        import visualization.live_training_dashboard
        importlib.reload(visualization.live_training_dashboard)

        from visualization.live_training_dashboard import LiveDashboard
        self.LiveDashboard = LiveDashboard

        self.root = MagicMock()

    def tearDown(self):
        self.patcher.stop()

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_metrics_update(self, mock_thread):
        # Prevent thread from starting
        mock_thread.return_value.start = MagicMock()

        dashboard = self.LiveDashboard(self.root)

        # Configure internal card mocks to capture updates
        # _make_card returns (value_label, sub_label)
        mock_wr_label = MagicMock()
        dashboard._card_wr = (mock_wr_label, MagicMock())

        mock_pnl_label = MagicMock()
        dashboard._card_pnl = (mock_pnl_label, MagicMock())

        # Sample Data
        dashboard.data = {
            'iteration': 10,
            'total_iterations': 100,
            'elapsed_seconds': 3600,
            'states_learned': 50,
            'high_confidence_states': 10,
            'total_pnl': 70.0,
            'today_pnl': 70.0,
            'cumulative_win_rate': 66.7,
            'today_win_rate': 0.667,
            'trades': [
                {'result': 'WIN', 'pnl': 100.0},
                {'result': 'LOSS', 'pnl': -50.0},
                {'result': 'WIN', 'pnl': 20.0}
            ]
        }
        # Precompute values that LiveDashboard calculates from trades
        dashboard.data['total_pnl'] = 70.0
        dashboard.data['cumulative_win_rate'] = 66.7

        # Call refresh
        dashboard.refresh_dashboard()

        # Check Win Rate
        # refresh_dashboard calls .config(text=..., fg=...)
        # We need to verify kwargs

        # Check that it was called
        self.assertTrue(mock_wr_label.config.called, "Win Rate label config not called")
        call_kwargs = mock_wr_label.config.call_args.kwargs
        # Fallback if positional args used
        if not call_kwargs and mock_wr_label.config.call_args.args:
             # This shouldn't happen with current implementation
             pass

        self.assertEqual(call_kwargs.get('text'), "66.7%")
        self.assertEqual(call_kwargs.get('fg'), "#00ff00")

        # Check P&L
        self.assertTrue(mock_pnl_label.config.called, "P&L label config not called")
        call_kwargs_pnl = mock_pnl_label.config.call_args.kwargs
        self.assertEqual(call_kwargs_pnl.get('text'), "$70.00")
        self.assertEqual(call_kwargs_pnl.get('fg'), "#00ff00")

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_metrics_update_negative(self, mock_thread):
        # Test negative scenarios (Losses)
        mock_thread.return_value.start = MagicMock()
        dashboard = LiveDashboard(self.root)

        mock_wr_label = MagicMock()
        dashboard._card_wr = (mock_wr_label, MagicMock())

        mock_pnl_label = MagicMock()
        dashboard._card_pnl = (mock_pnl_label, MagicMock())

        dashboard.data = {
            'iteration': 10,
            'total_pnl': -150.0,
            'today_pnl': -150.0,
            'cumulative_win_rate': 0.0,
            'today_win_rate': 0.0,
            'trades': [
                {'result': 'LOSS', 'pnl': -100.0},
                {'result': 'LOSS', 'pnl': -50.0}
            ]
        }

        dashboard.refresh_dashboard()

        # Win Rate: 0% -> Red
        call_kwargs_wr = mock_wr_label.config.call_args.kwargs
        self.assertEqual(call_kwargs_wr.get('text'), "0.0%")
        self.assertEqual(call_kwargs_wr.get('fg'), "#ff4444")

        # P&L: -150.0 -> Red
        call_kwargs_pnl = mock_pnl_label.config.call_args.kwargs
        self.assertEqual(call_kwargs_pnl.get('text'), "$-150.00")
        self.assertEqual(call_kwargs_pnl.get('fg'), "#ff4444")

if __name__ == '__main__':
    unittest.main()
