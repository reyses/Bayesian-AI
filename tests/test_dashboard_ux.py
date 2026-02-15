import unittest
import sys
import os
from unittest.mock import MagicMock, patch
import importlib

class TestDashboardUX(unittest.TestCase):
    def setUp(self):
        # Create Mocks FRESH for each test
        self.mock_tk = MagicMock()
        self.mock_ttk = MagicMock()

        def create_mock_label(*args, **kwargs):
            m = MagicMock()
            # Ensure bind is a mock
            m.bind = MagicMock()
            m.config = MagicMock()
            m.pack = MagicMock()
            m.grid = MagicMock()
            return m

        # Set it on the mock_ttk object
        self.mock_ttk.Label = create_mock_label
        # Also mock Scrollbar
        self.mock_ttk.Scrollbar = MagicMock()

        # Set it on mock_tk.Label too (because LiveDashboard uses tk.Label for cards)
        self.mock_tk.Label = MagicMock(side_effect=create_mock_label)

        # Mock Text widget
        self.mock_tk.Text = MagicMock()

        # IMPORTANT: Link mock_tk.ttk to mock_ttk!
        self.mock_tk.ttk = self.mock_ttk

        # Patch dict for sys.modules
        self.modules_patcher = patch.dict(sys.modules, {
            'tkinter': self.mock_tk,
            'tkinter.ttk': self.mock_ttk,
            'tkinter.messagebox': MagicMock(),
            'matplotlib': MagicMock(),
            'matplotlib.pyplot': MagicMock(),
            'matplotlib.backends': MagicMock(),
            'matplotlib.backends.backend_tkagg': MagicMock(),
            'matplotlib.figure': MagicMock(),
            'matplotlib.dates': MagicMock(),
            'matplotlib.style': MagicMock(),
        })
        self.modules_patcher.start()

        # Import module inside patched environment
        import visualization.live_training_dashboard
        importlib.reload(visualization.live_training_dashboard)
        self.dashboard_module = visualization.live_training_dashboard
        self.LiveDashboard = self.dashboard_module.LiveDashboard

        self.root = MagicMock()

    def tearDown(self):
        self.modules_patcher.stop()

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_tooltips_presence(self, mock_thread):
        mock_thread.return_value.start = MagicMock()
        dashboard = self.LiveDashboard(self.root)

        # Verify ids are different (Sanity check)
        # Note: _card_states is a tuple (val_label, sub_label)
        # We compare the value labels
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
        dashboard = self.LiveDashboard(self.root)

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
        self.mock_ttk.Scrollbar.reset_mock()

        dashboard = self.LiveDashboard(self.root)

        # Verify Scrollbar was instantiated
        self.assertTrue(self.mock_ttk.Scrollbar.called, "Scrollbar should be instantiated")

        # Verify text widget configured with yscrollcommand
        # dashboard.txt_log is the mock instance
        configure_calls = dashboard.txt_log.configure.call_args_list
        has_yscroll = False
        for call in configure_calls:
            if 'yscrollcommand' in call.kwargs:
                has_yscroll = True
                break

        self.assertTrue(has_yscroll, "Text widget should be configured with yscrollcommand")

        # Verify Scrollbar linked to text widget yview
        scrollbar_calls = self.mock_ttk.Scrollbar.call_args_list
        found_command = False
        for call in scrollbar_calls:
            if 'command' in call.kwargs and call.kwargs['command'] == dashboard.txt_log.yview:
                found_command = True
                break

        self.assertTrue(found_command, "Scrollbar should be linked to txt_log.yview")

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_shortcuts_binding(self, mock_thread):
        mock_thread.return_value.start = MagicMock()
        dashboard = self.LiveDashboard(self.root)

        # Check bindings on root
        bind_calls = self.root.bind.call_args_list
        bound_keys = [call[0][0] for call in bind_calls]

        expected_keys = ['<space>', 'p', 'P', 's', 'S', 'x', 'X']

        for key in expected_keys:
            self.assertIn(key, bound_keys, f"Key {key} should be bound")

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_log_read_only(self, mock_thread):
        mock_thread.return_value.start = MagicMock()

        # Instantiate dashboard
        dashboard = self.LiveDashboard(self.root)

        # Check call args of tk.Text constructor
        # self.mock_tk is the mock object for 'tkinter'
        call_args = self.mock_tk.Text.call_args

        if call_args:
             self.assertEqual(call_args.kwargs.get('state'), 'disabled', "Log text widget should be initialized as disabled")

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_toggle_pause(self, mock_thread):
        mock_thread.return_value.start = MagicMock()
        dashboard = self.LiveDashboard(self.root)

        if not hasattr(dashboard, 'toggle_pause'):
            self.fail("toggle_pause method not found")

        # Mock pause/resume methods
        dashboard.pause_training = MagicMock()
        dashboard.resume_training = MagicMock()

        # Case 1: Running -> Pause
        dashboard.remote_status = "RUNNING"
        dashboard.toggle_pause()
        dashboard.pause_training.assert_called_once()
        dashboard.resume_training.assert_not_called()

        dashboard.pause_training.reset_mock()
        dashboard.resume_training.reset_mock()

        # Case 2: Paused -> Resume
        dashboard.remote_status = "PAUSED"
        dashboard.toggle_pause()
        dashboard.resume_training.assert_called_once()
        dashboard.pause_training.assert_not_called()

if __name__ == '__main__':
    unittest.main()
