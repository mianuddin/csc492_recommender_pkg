import os
from unittest import TestSuite
from .metrics import TestMetrics


test_cases = [TestMetrics]


def load_tests(loader, tests, pattern):
    suite = TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite
