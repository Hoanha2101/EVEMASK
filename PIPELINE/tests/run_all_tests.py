"""
Script to run all unit tests for EVEMASK Pipeline system.
"""

import unittest
import os
import sys

if __name__ == '__main__':
    # Discover and run all tests in the current directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    suite = unittest.defaultTestLoader.discover(test_dir, pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful()) 