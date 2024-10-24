import unittest
from bcipy.task.control.query import best_selection


class TestQueryMechanisms(unittest.TestCase):
    def test_best_selection(self):
        list_el = ["A", "E", "I", "O", "U"]
        values = [0.4, 0.1, 0.25, 0.1, 0.15]
        len_query = 3
        self.assertEqual(["A", "I", "U"], best_selection(list_el, values, len_query))

        list_el = ["A", "E", "I", "O", "U"]
        values = [0.15, 0.4, 0.1, 0.25, 0.1]
        len_query = 3
        self.assertEqual(["E", "O", "A"], best_selection(list_el, values, len_query))

        list_el = ["A", "E", "I", "O", "U"]
        values = [0.1, 0.2, 0.2, 0.2, 0.2]
        len_query = 3
        self.assertEqual(["E", "I", "O"], best_selection(list_el, values, len_query))


if __name__ == '__main__':
    unittest.main()
