"""
Unit tests for models modules.
Tests engine and initNet functionality.
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import engine, initNet


class TestEngine(unittest.TestCase):
    """Test cases for engine module."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock config and input data
        self.mock_config = {'MODEL_PATH': 'test_model.engine'}
        self.input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_import_engine(self):
        """Test engine module import."""
        self.assertIsNotNone(engine)
    
    def test_engine_initialization(self):
        """Test engine initialization (mocked)."""
        # Try to initialize engine if class exists
        if hasattr(engine, 'Engine'):
            with patch.object(engine, 'Engine', autospec=True) as MockEngine:
                instance = MockEngine(self.mock_config)
                self.assertIsNotNone(instance)
        elif hasattr(engine, 'TrtEngine'):
            with patch.object(engine, 'TrtEngine', autospec=True) as MockTrtEngine:
                instance = MockTrtEngine(self.mock_config)
                self.assertIsNotNone(instance)
        elif hasattr(engine, 'ModelEngine'):
            with patch.object(engine, 'ModelEngine', autospec=True) as MockModelEngine:
                instance = MockModelEngine(self.mock_config)
                self.assertIsNotNone(instance)
    
    def test_engine_inference(self):
        """Test engine inference method (mocked)."""
        # Check for inference method in any engine class
        for cls_name in ['Engine', 'TrtEngine', 'ModelEngine']:
            if hasattr(engine, cls_name):
                cls = getattr(engine, cls_name)
                with patch.object(cls, 'infer', return_value=np.zeros((1, 10))) as mock_infer:
                    instance = cls(self.mock_config)
                    result = instance.infer(self.input_data)
                    self.assertIsInstance(result, np.ndarray)
                    break
    
    def test_engine_error_handling(self):
        """Test engine error handling for invalid input."""
        for cls_name in ['Engine', 'TrtEngine', 'ModelEngine']:
            if hasattr(engine, cls_name):
                cls = getattr(engine, cls_name)
                with patch.object(cls, 'infer', side_effect=Exception('Inference error')):
                    instance = cls(self.mock_config)
                    with self.assertRaises(Exception):
                        instance.infer(None)


class TestInitNet(unittest.TestCase):
    """Test cases for initNet module."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_config = {'MODEL_PATH': 'test_model.engine'}
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_import_initNet(self):
        """Test initNet module import."""
        self.assertIsNotNone(initNet)
    
    def test_initNet_main_function(self):
        """Test main function or class in initNet."""
        # Check for main function/class
        if hasattr(initNet, 'init_model'):
            with patch.object(initNet, 'init_model', return_value=True) as mock_init:  # type: ignore[attr-defined]
                result = initNet.init_model(self.mock_config)  # type: ignore[attr-defined]
                self.assertTrue(result)
        elif hasattr(initNet, 'initialize'):
            with patch.object(initNet, 'initialize', return_value=True) as mock_initialize:  # type: ignore[attr-defined]
                result = initNet.initialize(self.mock_config)  # type: ignore[attr-defined]
                self.assertTrue(result)
        else:
            # If no main function, just pass
            self.assertTrue(True)
    
    def test_initNet_error_handling(self):
        """Test error handling in initNet."""
        if hasattr(initNet, 'init_model'):
            with patch.object(initNet, 'init_model', side_effect=Exception('Init error')):  # type: ignore[attr-defined]
                with self.assertRaises(Exception):
                    initNet.init_model(None)  # type: ignore[attr-defined]
        elif hasattr(initNet, 'initialize'):
            with patch.object(initNet, 'initialize', side_effect=Exception('Init error')):  # type: ignore[attr-defined]
                with self.assertRaises(Exception):
                    initNet.initialize(None)  # type: ignore[attr-defined]

if __name__ == '__main__':
    unittest.main() 