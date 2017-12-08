import unittest
import similarity_measures


class TestSimilarityMeasure(unittest.TestCase):
    def test_correlation_coefficient(self):
        a = [1,4,6]
        b = [1,2,3]
        self.assertAlmostEqual(similarity_measures.correlation_coefficient(a,b), 0.99339926779878274)
