"""
Script to run all unit tests for the EVEMASK Pipeline system.

This script discovers and executes all test modules in the tester package, providing a unified test entry point for the entire system.

Author: EVEMASK Team
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