import unittest
import sys
import os
import shutil
from unittest.mock import MagicMock, patch
import importlib

class TestDashboardControls(unittest.TestCase):
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

        self.test_dir = "tests/test_dashboard_data"
        os.makedirs(self.test_dir, exist_ok=True)

        # Mock the dashboard instance
        self.root = MagicMock()

    def tearDown(self):
        self.patcher.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_pause_resume_stop(self, mock_thread):
        # Prevent thread from starting
        mock_thread.return_value.start = MagicMock()

        dashboard = self.LiveDashboard(self.root)

        # Overwrite paths to use test directory
        dashboard.training_dir = self.test_dir
        dashboard.pause_file = os.path.join(self.test_dir, 'PAUSE')
        dashboard.stop_file = os.path.join(self.test_dir, 'STOP')
        dashboard.json_path = os.path.join(self.test_dir, 'training_progress.json')

        # Test Pause
        dashboard.pause_training()
        self.assertTrue(os.path.exists(dashboard.pause_file))
        with open(dashboard.pause_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, 'PAUSE')

        # Test Resume
        dashboard.resume_training()
        self.assertFalse(os.path.exists(dashboard.pause_file))

        # Test Stop (mock messagebox to say yes)
        # Note: We mock self.mock_messagebox because that's what LiveDashboard sees
        self.mock_messagebox.askyesno.return_value = True

        dashboard.stop_training()

        self.assertTrue(os.path.exists(dashboard.stop_file))
        with open(dashboard.stop_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, 'STOP')

if __name__ == '__main__':
    unittest.main()
