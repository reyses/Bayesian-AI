import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.databento_loader import DatabentoLoader
from training.orchestrator import TrainingOrchestrator

class TestDatabentoLoading:
    @pytest.fixture
    def mock_databento(self):
        with patch('training.databento_loader.db') as mock_db:
            yield mock_db

    @patch('os.path.exists')
    def test_databento_loader_structure(self, mock_exists, mock_databento):
        mock_exists.return_value = True
        # Setup mock return data
        mock_store = MagicMock()
        mock_db_df = pd.DataFrame({
            'ts_event': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:00:01']),
            'price': [100.0, 101.0],
            'size': [10, 20],
            'action': ['T', 'T'],
            'other': [1, 2]
        })
        mock_store.to_df.return_value = mock_db_df
        mock_databento.DBNStore.from_file.return_value = mock_store

        # Call loader
        df = DatabentoLoader.load_data("dummy.dbn")

        # Verify calls
        mock_databento.DBNStore.from_file.assert_called_with("dummy.dbn")

        # Verify result
        assert 'timestamp' in df.columns
        assert 'price' in df.columns
        assert 'volume' in df.columns
        assert 'type' in df.columns

        # Verify conversions
        assert df.iloc[0]['volume'] == 10
        assert df.iloc[0]['type'] == 'T'
        assert isinstance(df.iloc[0]['timestamp'], float)

    @patch('os.path.exists')
    def test_databento_loader_filtering(self, mock_exists, mock_databento):
        mock_exists.return_value = True
        # Setup mock return data with mixed actions
        mock_store = MagicMock()
        mock_db_df = pd.DataFrame({
            'ts_event': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:00:01']),
            'price': [100.0, 101.0],
            'size': [10, 20],
            'action': ['T', 'A'] # Trade and Add
        })
        mock_store.to_df.return_value = mock_db_df
        mock_databento.DBNStore.from_file.return_value = mock_store

        # Call loader with filtering (default)
        df = DatabentoLoader.load_data("dummy.dbn")
        assert len(df) == 1
        assert df.iloc[0]['type'] == 'T'

        # Call loader without filtering
        mock_store.to_df.return_value = mock_db_df # Reset iterator if needed, but it's a mock
        df_all = DatabentoLoader.load_data("dummy.dbn", filter_trades=False)
        assert len(df_all) == 2

    @patch('training.orchestrator.DatabentoLoader')
    @patch('training.orchestrator.pd.read_parquet')
    def test_orchestrator_integration(self, mock_read_parquet, mock_loader):
        # Mock loader return
        mock_df = pd.DataFrame({'timestamp': [], 'price': [], 'volume': [], 'type': []})
        mock_loader.load_data.return_value = mock_df

        # Mock parquet return
        mock_read_parquet.return_value = mock_df

        # Test .dbn loading
        # Note: TrainingOrchestrator init requires asset_ticker that exists in SYMBOL_MAP ("NQ")
        orch_dbn = TrainingOrchestrator("NQ", data_path="test.dbn", use_gpu=False)
        mock_loader.load_data.assert_called_with("test.dbn")
        mock_read_parquet.assert_not_called()

        # Test .parquet loading
        mock_loader.load_data.reset_mock()
        mock_read_parquet.reset_mock()

        orch_parquet = TrainingOrchestrator("NQ", data_path="test.parquet", use_gpu=False)
        mock_read_parquet.assert_called_with("test.parquet")
        mock_loader.load_data.assert_not_called()
