import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Mock setup
mock_tk = MagicMock()
mock_ttk = MagicMock()

def create_mock_label(*args, **kwargs):
    m = MagicMock()
    return m

# Set it on the mock_ttk object
mock_ttk.Label = create_mock_label
mock_tk.Label = MagicMock(side_effect=create_mock_label)
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
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Now import the dashboard
from visualization.live_training_dashboard import LiveDashboard, COLOR_CARD_BG

class TestDashboardZebra(unittest.TestCase):
    def setUp(self):
        self.root = MagicMock()

    @patch('visualization.live_training_dashboard.threading.Thread')
    def test_zebra_striping(self, mock_thread):
        mock_thread.return_value.start = MagicMock()
        dashboard = LiveDashboard(self.root)

        # 1. Verify tag configuration
        # Check if tag_configure was called with 'oddrow'
        tree = dashboard.day_tree
        calls = tree.tag_configure.call_args_list

        found_oddrow_config = False
        for call in calls:
            # call.args[0] is the tag name
            if call.args and call.args[0] == 'oddrow':
                found_oddrow_config = True
                # Check background color in kwargs
                self.assertEqual(call.kwargs.get('background'), COLOR_CARD_BG)
                break

        self.assertTrue(found_oddrow_config, "Should configure 'oddrow' tag on Treeview with correct background color")

        # 2. Verify update_day_table applies tags correctly
        # Create dummy data
        data = {
            'day_summaries': [
                {'day': 1, 'pnl': 100, 'win_rate': 0.5},  # Index 0 (Even) -> No oddrow, Profit
                {'day': 2, 'pnl': -50, 'win_rate': 0.4},  # Index 1 (Odd) -> oddrow, Loss
                {'day': 3, 'pnl': 200, 'win_rate': 0.6},  # Index 2 (Even) -> No oddrow, Profit
                {'day': 4, 'pnl': 100, 'win_rate': 0.5},  # Index 3 (Odd) -> oddrow, Profit
            ]
        }

        # Reset mock calls to ignore initialization calls
        tree.insert.reset_mock()

        dashboard.update_day_table(data)

        # Check insert calls
        insert_calls = tree.insert.call_args_list
        self.assertEqual(len(insert_calls), 4, "Should insert 4 rows")

        # Row 0 (Even index) -> profit tag only
        # insert('', 'end', values=..., tags=...)
        call0 = insert_calls[0]
        tags0 = call0.kwargs.get('tags')
        self.assertIn('profit', tags0)
        self.assertNotIn('oddrow', tags0)

        # Row 1 (Odd index) -> loss tag + oddrow tag
        call1 = insert_calls[1]
        tags1 = call1.kwargs.get('tags')
        self.assertIn('loss', tags1)
        self.assertIn('oddrow', tags1)

        # Row 2 (Even index) -> profit tag only
        call2 = insert_calls[2]
        tags2 = call2.kwargs.get('tags')
        self.assertIn('profit', tags2)
        self.assertNotIn('oddrow', tags2)

        # Row 3 (Odd index) -> profit tag + oddrow tag
        call3 = insert_calls[3]
        tags3 = call3.kwargs.get('tags')
        self.assertIn('profit', tags3)
        self.assertIn('oddrow', tags3)

if __name__ == '__main__':
    unittest.main()
