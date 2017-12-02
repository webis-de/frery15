import unittest
import main2

class TestMain2(unittest.TestCase):
    def test_correlation_coefficient(self):
        a = [1,4,6]
        b = [1,2,3]
        self.assertAlmostEqual(main2.correlation_coefficient(a,b), 0.99339926779878274)