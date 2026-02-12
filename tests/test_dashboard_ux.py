import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Note: We rely on python -m unittest for path resolution now.

# Create Mocks
mock_tk = MagicMock()
mock_ttk = MagicMock()

def create_mock_label(*args, **kwargs):
    m = MagicMock()
    return m

# Set it on the mock_ttk object
mock_ttk.Label = create_mock_label

# Set it on mock_tk.Label too (because LiveDashboard uses tk.Label for cards)
mock_tk.Label = MagicMock(side_effect=create_mock_label)

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
        self.assertNotEqual(id(dashboard._card_states[0]), id(dashboard._card_progress[0]), "Labels should be different objects")

        def check_binding(widget, widget_name):
            calls = widget.bind.call_args_list
            if not calls:
                return False
            has_enter = any(call[0][0] == "<Enter>" for call in calls)
            return has_enter

        # These already have tooltips
        self.assertTrue(check_binding(dashboard._card_states[0], "lbl_states"), "lbl_states should have tooltip")
        self.assertTrue(check_binding(dashboard._card_states[1], "lbl_conf"), "lbl_conf should have tooltip")

        # These SHOULD have tooltips after my changes
        self.assertTrue(check_binding(dashboard._card_progress[0], "lbl_iter"), "lbl_iter should have tooltip")
        self.assertTrue(check_binding(dashboard._card_progress[1], "lbl_time"), "lbl_time should have tooltip")
        # lbl_eta is the same as lbl_time in implementation
        self.assertTrue(check_binding(dashboard._card_progress[1], "lbl_eta"), "lbl_eta should have tooltip")
        self.assertTrue(check_binding(dashboard._card_trades[0], "lbl_trades"), "lbl_trades should have tooltip")
        self.assertTrue(check_binding(dashboard._card_pnl[0], "lbl_pnl"), "lbl_pnl should have tooltip")
        self.assertTrue(check_binding(dashboard._card_wr[0], "lbl_wr"), "lbl_wr should have tooltip")

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
            with self.subTest(status=status):
                dashboard.remote_status = status
                # Reset mock to clear previous calls for clean checking
                dashboard.lbl_status.config.reset_mock()

                dashboard.update_gui()

                # Check the keyword arguments of the call to config
                # We expect one call to config since we reset the mock
                call_args = dashboard.lbl_status.config.call_args
                self.assertIsNotNone(call_args, f"config() not called for {status}")

                kwargs = call_args.kwargs
                # If config was called with positional args? Usually config(text=...)
                if not kwargs and call_args.args:
                     # This shouldn't happen based on code but handle defensively if needed
                     pass

                self.assertIn('text', kwargs, f"text argument missing in config call for {status}")
                called_text = kwargs['text']

                self.assertIn(icon, called_text, f"Icon {icon} for {status} not found in '{called_text}'")
                self.assertIn(status, called_text, f"Status text {status} not found in '{called_text}'")

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_log_scrollbar(self, mock_thread):
        mock_thread.return_value.start = MagicMock()

        # Reset mock to spy on creation
        mock_ttk.Scrollbar.reset_mock()

        dashboard = LiveDashboard(self.root)

        # Verify Scrollbar was instantiated
        self.assertTrue(mock_ttk.Scrollbar.called, "Scrollbar should be instantiated")

        # Verify text widget configured with yscrollcommand
        # dashboard.txt_log is the mock instance
        mock_scrollbar_instance = mock_ttk.Scrollbar.return_value
        dashboard.txt_log.configure.assert_any_call(yscrollcommand=mock_scrollbar_instance.set)

        # Verify Scrollbar linked to text widget yview
        scrollbar_calls = mock_ttk.Scrollbar.call_args_list
        found_command = False
        for call in scrollbar_calls:
            if 'command' in call.kwargs and call.kwargs['command'] == dashboard.txt_log.yview:
                found_command = True
                break

        self.assertTrue(found_command, "Scrollbar should be linked to txt_log.yview")

if __name__ == '__main__':
    unittest.main()
