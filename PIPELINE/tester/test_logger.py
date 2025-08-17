"""
Unit tests for EveMaskLogger module in the EVEMASK Pipeline system.

This module provides comprehensive tests for logging and display functionality, including FPS tracking, configuration display, and singleton pattern validation.

Author: EVEMASK Team
"""

import unittest
import sys
import os
import time
from unittest.mock import patch, MagicMock, call
from io import StringIO

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.logger import EveMaskLogger


class TestEveMaskLogger(unittest.TestCase):
    """Test cases for EveMaskLogger class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Reset singleton instance before each test
        EveMaskLogger._global_instance = EveMaskLogger()
        self.logger = EveMaskLogger()
        
        self.test_config = {
                            'CLASSES_NO_BLUR': [0],
                            'DELAY_TIME': 2,
                            'TARGET_FPS': 10,
                            'batch_size': 1,
                            'INPUT_SOURCE': 'udp://224.1.1.1:30122?pkt_size=1316',
                            'OUTPUT_TYPE': 'udp',
                            'OUTPUT_STREAM_URL_RTMP': None,
                            'OUTPUT_STREAM_URL_RTSP': None,
                            'OUTPUT_STREAM_URL_UDP': 'udp://@225.1.9.254:30133?pkt_size=1316',
                            'conf_threshold': 0.5,
                            'iou_threshold': 0.7,
                            'nc': 14,
                            'names': {
                                0: 'unbet',
                                1: 'betrivers',
                                2: 'fanduel',
                                3: 'betway',
                                4: 'caesars',
                                5: 'bally',
                                6: 'draftkings',
                                7: 'pointsbet',
                                8: 'bet365',
                                9: 'fanatics',
                                10: 'betparx',
                                11: 'betmgm',
                                12: 'gilariver',
                                13: 'casino',
                            },
                            'recognizeData_path': 'recognizeData',
                            'segment_model': {
                                'all_output_names': [
                                    'pred0',
                                    'pred1_0_0',
                                    'pred1_0_1',
                                    'pred1_0_2',
                                    'pred1_1',
                                    'pred1_2'
                                ],
                                'dynamic_factor': 3,
                                'get_to': 'cuda',
                                'input_names': ['input'],
                                'max_batch_size': 3,
                                'path': 'weights/trtPlans/yolov8_seg_aug_best_l_trimmed.trt'
                            },
                            'extract_model': {
                                'input_names': ['input'],
                                'len_emb': 256,
                                'max_batch_size': 32,
                                'output_names': ['output'],
                                'path': 'weights/trtPlans/SupConLoss_BBVGG16.trt'
                            }
                        }

    
    def tearDown(self):
        """Clean up after each test method."""
        EveMaskLogger._global_instance = EveMaskLogger()
    
    def test_initialization(self):
        """Test EveMaskLogger initialization."""
        self.assertEqual(self.logger.version, "1.0.0")
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
    
    def test_show_config(self):
        """Test configuration display functionality."""
        test_config = {
            'INPUT_SOURCE': 'camera',
            'OUTPUT_TYPE': 'display',
            'batch_size': 4,
            'TARGET_FPS': 30
        }
        # Just ensure it runs without error
        self.logger.show_config(test_config)

    def test_show_config_missing_keys(self):
        """Test configuration display with missing keys."""
        test_config = {'some_key': 'some_value'}
        # Just ensure it runs without error
        self.logger.show_config(test_config)

    def test_waiting_bar(self):
        """Test progress bar display functionality."""
        test_config = {'DELAY_TIME': 0.1}
        # Just ensure it runs without error
        self.logger.waiting_bar(test_config)

    def test_display_stream(self):
        """Test real-time stream statistics display."""
        # Set some test values
        self.logger.update_in_stream_fps(30.0)
        self.logger.update_out_stream_fps(25.0)
        self.logger.update_ai_fps(15.0)
        self.logger.update_number_out_frames(1000)
        self.logger.update_n_skip_frames(50)

        table = self.logger.display_stream(self.test_config, True, True, True)
        self.assertEqual(table.title, "🚀 EVEMASK STREAM LOGGER")
        # Get all cell values
        all_cells = [str(cell) for col in table.columns for cell in col._cells]
        self.assertIn("🎥 Input FPS", all_cells)
        self.assertIn("📤 Output FPS", all_cells)
        self.assertIn("🧠 AI FPS", all_cells)
        self.assertIn("🖼️ Frames Output", all_cells)
        self.assertIn("🕳️ Skipped Frames", all_cells)

    def test_display_stream_with_none_values(self):
        """Test display with None FPS values."""
        # Set invalid (negative) values to simulate 'None' handling
        self.logger.in_stream_fps = -1
        self.logger.out_stream_fps = -1
        self.logger.ai_fps = -1

        table = self.logger.display_stream(self.test_config, True, True, True)
        all_cells = [str(cell) for col in table.columns for cell in col._cells]
        self.assertIn("🎥 Input FPS", all_cells)
        self.assertIn("📤 Output FPS", all_cells)
        self.assertIn("🧠 AI FPS", all_cells)

    def test_display_logo(self):
        """Test logo display functionality."""
        # Just ensure it runs without error
        self.logger.display_logo()
    
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
        self.assertEqual(instance.version, "1.0.0")
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
    
    def test_display_stream_formatting(self):
        """Test that display_stream formats output correctly."""
        # Set test values
        self.logger.update_in_stream_fps(30.123)
        self.logger.update_out_stream_fps(25.456)
        self.logger.update_ai_fps(15.789)
        self.logger.update_number_out_frames(12345)
        self.logger.update_n_skip_frames(678)

        table = self.logger.display_stream(self.test_config, True, True, True)
        all_cells = [str(cell) for col in table.columns for cell in col._cells]
        self.assertTrue(any("30.1" in cell for cell in all_cells))
        self.assertTrue(any("25.5" in cell for cell in all_cells))
        self.assertTrue(any("15.8" in cell for cell in all_cells))
        self.assertIn("12345", all_cells)
        self.assertIn("678", all_cells)


if __name__ == '__main__':
    unittest.main() 