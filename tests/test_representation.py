import unittest
import representation
import numpy as np


class TestRepresentationSpaces(unittest.TestCase):
    def test_avg_marks(self):
        sentence = 'Lorem, ipsum: est!'
        correct_vector = [[1], [0], [1], [0], [0], [1], [0]]

        self.assertEqual(len([counted for counted, correct in zip(representation.avg_marks(sentence), correct_vector) if
                              counted == correct]), len(correct_vector))
