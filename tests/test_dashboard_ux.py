import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create Mocks
mock_tk = MagicMock()
mock_ttk = MagicMock()

def create_mock_label(*args, **kwargs):
    m = MagicMock()
    return m

# Set it on the mock_ttk object
mock_ttk.Label = create_mock_label

# IMPORTANT: Link mock_tk.ttk to mock_ttk!
mock_tk.ttk = mock_ttk

# Assign to sys.modules
sys.modules['tkinter'] = mock_tk
sys.modules['tkinter.ttk'] = mock_ttk
sys.modules['tkinter.messagebox'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.backends'] = MagicMock()
sys.modules['matplotlib.backends.backend_tkagg'] = MagicMock()
sys.modules['matplotlib.figure'] = MagicMock()
sys.modules['matplotlib.dates'] = MagicMock()
sys.modules['matplotlib.style'] = MagicMock()

# Now import the dashboard
from visualization.live_training_dashboard import LiveDashboard, Tooltip

class TestDashboardUX(unittest.TestCase):
    def setUp(self):
        self.root = MagicMock()

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_tooltips_presence(self, mock_thread):
        mock_thread.return_value.start = MagicMock()
        dashboard = LiveDashboard(self.root)

        # Verify ids are different (Sanity check)
        self.assertNotEqual(id(dashboard.lbl_states), id(dashboard.lbl_iter), "Labels should be different objects")

        def check_binding(widget, widget_name):
            calls = widget.bind.call_args_list
            if not calls:
                return False
            has_enter = any(call[0][0] == "<Enter>" for call in calls)
            return has_enter

        # These already have tooltips
        self.assertTrue(check_binding(dashboard.lbl_states, "lbl_states"), "lbl_states should have tooltip")
        self.assertTrue(check_binding(dashboard.lbl_conf, "lbl_conf"), "lbl_conf should have tooltip")

        # These SHOULD have tooltips after my changes
        self.assertTrue(check_binding(dashboard.lbl_iter, "lbl_iter"), "lbl_iter should have tooltip")
        self.assertTrue(check_binding(dashboard.lbl_time, "lbl_time"), "lbl_time should have tooltip")
        self.assertTrue(check_binding(dashboard.lbl_eta, "lbl_eta"), "lbl_eta should have tooltip")
        self.assertTrue(check_binding(dashboard.lbl_trades, "lbl_trades"), "lbl_trades should have tooltip")
        self.assertTrue(check_binding(dashboard.lbl_pnl, "lbl_pnl"), "lbl_pnl should have tooltip")
        self.assertTrue(check_binding(dashboard.lbl_wr, "lbl_wr"), "lbl_wr should have tooltip")

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_status_icons(self, mock_thread):
        mock_thread.return_value.start = MagicMock()
        dashboard = LiveDashboard(self.root)

        # Test RUNNING
        dashboard.remote_status = "RUNNING"
        dashboard.update_gui()

        found = False
        for call in dashboard.lbl_status.config.call_args_list:
             if 'text' in call[1] and "🟢" in call[1]['text']:
                 found = True
                 break
        self.assertTrue(found, "Status icon 🟢 should be present for RUNNING")

        # Test PAUSED
        dashboard.remote_status = "PAUSED"
        dashboard.update_gui()
        found = False
        for call in dashboard.lbl_status.config.call_args_list:
             if 'text' in call[1] and "⏸️" in call[1]['text']:
                 found = True
                 break
        self.assertTrue(found, "Status icon ⏸️ should be present for PAUSED")

        # Test STOPPED
        dashboard.remote_status = "STOPPED"
        dashboard.update_gui()
        found = False
        for call in dashboard.lbl_status.config.call_args_list:
             if 'text' in call[1] and "🛑" in call[1]['text']:
                 found = True
                 break
        self.assertTrue(found, "Status icon 🛑 should be present for STOPPED")

if __name__ == '__main__':
    unittest.main()
