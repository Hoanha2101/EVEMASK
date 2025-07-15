"""
Unit tests for EveMaskLogger module.
Tests logging and display functionality for EVEMASK pipeline.
"""

import unittest
import sys
import os
import time
from unittest.mock import patch, MagicMock
from io import StringIO

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from logger.logger import EveMaskLogger


class TestEveMaskLogger(unittest.TestCase):
    """Test cases for EveMaskLogger class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Reset singleton instance before each test
        EveMaskLogger._global_instance = None
        self.logger = EveMaskLogger()
    
    def tearDown(self):
        """Clean up after each test method."""
        # Reset singleton instance after each test
        EveMaskLogger._global_instance = None
        self.logger = None
    
    def test_initialization(self):
        """Test EveMaskLogger initialization."""
        self.assertEqual(self.logger.version, "2.0")
        self.assertEqual(self.logger.in_stream_fps, 0)
        self.assertEqual(self.logger.out_stream_fps, 0)
        self.assertEqual(self.logger.ai_fps, 0)
        self.assertEqual(self.logger.number_out_frames, 0)
        self.assertEqual(self.logger.n_skip_frames, 0)
    
    def test_update_in_stream_fps(self):
        """Test updating input stream FPS."""
        test_fps = 30.5
        self.logger.update_in_stream_fps(test_fps)
        self.assertEqual(self.logger.in_stream_fps, test_fps)
    
    def test_update_out_stream_fps(self):
        """Test updating output stream FPS."""
        test_fps = 25.0
        self.logger.update_out_stream_fps(test_fps)
        self.assertEqual(self.logger.out_stream_fps, test_fps)
    
    def test_update_ai_fps(self):
        """Test updating AI processing FPS."""
        test_fps = 15.7
        self.logger.update_ai_fps(test_fps)
        self.assertEqual(self.logger.ai_fps, test_fps)
    
    def test_update_number_out_frames(self):
        """Test updating number of output frames."""
        test_number = 1000
        self.logger.update_number_out_frames(test_number)
        self.assertEqual(self.logger.number_out_frames, test_number)
    
    def test_update_n_skip_frames(self):
        """Test updating number of skipped frames."""
        test_skip = 50
        self.logger.update_n_skip_frames(test_skip)
        self.assertEqual(self.logger.n_skip_frames, test_skip)
    
    @patch('builtins.print')
    def test_show_config(self, mock_print):
        """Test configuration display functionality."""
        test_config = {
            'INPUT_SOURCE': 'camera',
            'OUTPUT_TYPE': 'display',
            'batch_size': 4,
            'TARGET_FPS': 30
        }
        
        self.logger.show_config(test_config)
        
        # Check that print was called with expected messages
        expected_calls = [
            unittest.mock.call("✅ Configuration loaded"),
            unittest.mock.call("📥 Input source : camera"),
            unittest.mock.call("📤 Output type  : display"),
            unittest.mock.call("📦 Batch size   : 4"),
            unittest.mock.call("🎯 Target FPS   : 30"),
            unittest.mock.call("✅ All components initialized successfully")
        ]
        
        mock_print.assert_has_calls(expected_calls)
    
    @patch('builtins.print')
    def test_show_config_missing_keys(self, mock_print):
        """Test configuration display with missing keys."""
        test_config = {'some_key': 'some_value'}
        
        self.logger.show_config(test_config)
        
        # Should handle missing keys gracefully
        expected_calls = [
            unittest.mock.call("✅ Configuration loaded"),
            unittest.mock.call("📥 Input source : Not specified"),
            unittest.mock.call("📤 Output type  : Not specified"),
            unittest.mock.call("📦 Batch size   : Not specified"),
            unittest.mock.call("🎯 Target FPS   : Not specified"),
            unittest.mock.call("✅ All components initialized successfully")
        ]
        
        mock_print.assert_has_calls(expected_calls)
    
    @patch('sys.stdout.write')
    @patch('sys.stdout.flush')
    @patch('time.sleep')
    def test_waiting_bar(self, mock_sleep, mock_flush, mock_write):
        """Test progress bar display functionality."""
        test_config = {'DELAY_TIME': 0.1}
        
        self.logger.waiting_bar(test_config)
        
        # Check that write and flush were called
        self.assertGreater(mock_write.call_count, 0)
        self.assertGreater(mock_flush.call_count, 0)
        
        # Check that sleep was called for each step
        self.assertGreater(mock_sleep.call_count, 0)
    
    @patch('sys.stdout.write')
    @patch('sys.stdout.flush')
    def test_display_stream(self, mock_flush, mock_write):
        """Test real-time stream statistics display."""
        # Set some test values
        self.logger.update_in_stream_fps(30.0)
        self.logger.update_out_stream_fps(25.0)
        self.logger.update_ai_fps(15.0)
        self.logger.update_number_out_frames(1000)
        self.logger.update_n_skip_frames(50)
        
        self.logger.display_stream()
        
        # Check that write was called (for screen clearing and display)
        self.assertGreater(mock_write.call_count, 0)
        self.assertGreater(mock_flush.call_count, 0)
    
    @patch('sys.stdout.write')
    @patch('sys.stdout.flush')
    def test_display_stream_with_none_values(self, mock_flush, mock_write):
        """Test display with None FPS values."""
        # Set None values
        self.logger.in_stream_fps = None
        self.logger.out_stream_fps = None
        self.logger.ai_fps = None
        
        self.logger.display_stream()
        
        # Should handle None values gracefully
        self.assertGreater(mock_write.call_count, 0)
        self.assertGreater(mock_flush.call_count, 0)
    
    @patch('builtins.print')
    def test_display_logo(self, mock_print):
        """Test logo display functionality."""
        self.logger.display_logo()
        
        # Should call print at least once for the logo
        self.assertGreater(mock_print.call_count, 0)
    
    def test_singleton_pattern(self):
        """Test singleton pattern implementation."""
        instance1 = EveMaskLogger.get_instance()
        instance2 = EveMaskLogger.get_instance()
        
        # Should return the same instance
        self.assertIs(instance1, instance2)
    
    def test_singleton_initialization(self):
        """Test singleton initialization with default values."""
        instance = EveMaskLogger.get_instance()
        
        # Should have default values
        self.assertEqual(instance.version, "2.0")
        self.assertEqual(instance.in_stream_fps, 0)
        self.assertEqual(instance.out_stream_fps, 0)
        self.assertEqual(instance.ai_fps, 0)
    
    def test_fps_updates_consistency(self):
        """Test that FPS updates maintain consistency."""
        # Update all FPS values
        self.logger.update_in_stream_fps(30.0)
        self.logger.update_out_stream_fps(25.0)
        self.logger.update_ai_fps(15.0)
        
        # Verify all values are set correctly
        self.assertEqual(self.logger.in_stream_fps, 30.0)
        self.assertEqual(self.logger.out_stream_fps, 25.0)
        self.assertEqual(self.logger.ai_fps, 15.0)
        
        # Update with different values
        self.logger.update_in_stream_fps(60.0)
        self.logger.update_out_stream_fps(50.0)
        self.logger.update_ai_fps(30.0)
        
        # Verify new values
        self.assertEqual(self.logger.in_stream_fps, 60.0)
        self.assertEqual(self.logger.out_stream_fps, 50.0)
        self.assertEqual(self.logger.ai_fps, 30.0)
    
    def test_frame_count_updates(self):
        """Test frame count update functionality."""
        # Update frame counts
        self.logger.update_number_out_frames(100)
        self.logger.update_n_skip_frames(10)
        
        # Verify values
        self.assertEqual(self.logger.number_out_frames, 100)
        self.assertEqual(self.logger.n_skip_frames, 10)
        
        # Update with larger values
        self.logger.update_number_out_frames(1000)
        self.logger.update_n_skip_frames(100)
        
        # Verify new values
        self.assertEqual(self.logger.number_out_frames, 1000)
        self.assertEqual(self.logger.n_skip_frames, 100)
    
    def test_negative_fps_handling(self):
        """Test handling of negative FPS values."""
        # Test negative FPS values
        self.logger.update_in_stream_fps(-1.0)
        self.logger.update_out_stream_fps(-5.5)
        self.logger.update_ai_fps(-10.0)
        
        # Should accept negative values (though not ideal)
        self.assertEqual(self.logger.in_stream_fps, -1.0)
        self.assertEqual(self.logger.out_stream_fps, -5.5)
        self.assertEqual(self.logger.ai_fps, -10.0)
    
    def test_negative_frame_count_handling(self):
        """Test handling of negative frame count values."""
        # Test negative frame counts
        self.logger.update_number_out_frames(-10)
        self.logger.update_n_skip_frames(-5)
        
        # Should accept negative values
        self.assertEqual(self.logger.number_out_frames, -10)
        self.assertEqual(self.logger.n_skip_frames, -5)
    
    def test_zero_values_handling(self):
        """Test handling of zero values."""
        # Test zero values
        self.logger.update_in_stream_fps(0.0)
        self.logger.update_out_stream_fps(0.0)
        self.logger.update_ai_fps(0.0)
        self.logger.update_number_out_frames(0)
        self.logger.update_n_skip_frames(0)
        
        # Should handle zero values correctly
        self.assertEqual(self.logger.in_stream_fps, 0.0)
        self.assertEqual(self.logger.out_stream_fps, 0.0)
        self.assertEqual(self.logger.ai_fps, 0.0)
        self.assertEqual(self.logger.number_out_frames, 0)
        self.assertEqual(self.logger.n_skip_frames, 0)
    
    def test_large_values_handling(self):
        """Test handling of large values."""
        # Test large values
        large_fps = 999999.99
        large_count = 999999999
        
        self.logger.update_in_stream_fps(large_fps)
        self.logger.update_number_out_frames(large_count)
        
        # Should handle large values correctly
        self.assertEqual(self.logger.in_stream_fps, large_fps)
        self.assertEqual(self.logger.number_out_frames, large_count)
    
    @patch('sys.stdout.write')
    @patch('sys.stdout.flush')
    def test_display_stream_formatting(self, mock_flush, mock_write):
        """Test that display_stream formats output correctly."""
        # Set test values
        self.logger.update_in_stream_fps(30.123)
        self.logger.update_out_stream_fps(25.456)
        self.logger.update_ai_fps(15.789)
        self.logger.update_number_out_frames(12345)
        self.logger.update_n_skip_frames(678)
        
        self.logger.display_stream()
        
        # Check that write was called multiple times (for each line)
        self.assertGreater(mock_write.call_count, 5)
        
        # Check that flush was called
        self.assertGreater(mock_flush.call_count, 0)


if __name__ == '__main__':
    unittest.main() 