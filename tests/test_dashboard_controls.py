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

class TestDashboardControls(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_dashboard_data"
        os.makedirs(self.test_dir, exist_ok=True)

        # Mock the dashboard instance
        self.root = MagicMock()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_pause_resume_stop(self, mock_thread):
        # Prevent thread from starting
        mock_thread.return_value.start = MagicMock()

        dashboard = LiveDashboard(self.root)

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
        with patch('visualization.live_training_dashboard.messagebox.askyesno', return_value=True):
            dashboard.stop_training()
        self.assertTrue(os.path.exists(dashboard.stop_file))
        with open(dashboard.stop_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, 'STOP')

if __name__ == '__main__':
    unittest.main()
