import unittest

from bcipy.task.control.query import (NBestStimuliAgent, RandomStimuliAgent,
                                      best_selection)


class TestQueryMechanisms(unittest.TestCase):
    """Tests for query functions"""

    def test_best_selection(self):
        """Test best_selection function"""
        self.assertEqual(["A", "I", "U"],
                         best_selection(
                             selection_elements=["A", "E", "I", "O", "U"],
                             val=[0.4, 0.1, 0.25, 0.1, 0.15],
                             len_query=3))

        self.assertEqual(["E", "O", "A"],
                         best_selection(
                             selection_elements=["A", "E", "I", "O", "U"],
                             val=[0.15, 0.4, 0.1, 0.25, 0.1],
                             len_query=3))

        self.assertEqual(["E", "I", "O"],
                         best_selection(
                             selection_elements=["A", "E", "I", "O", "U"],
                             val=[0.1, 0.2, 0.2, 0.2, 0.2],
                             len_query=3))

        self.assertEqual(
            ["A", "E", "I"],
            best_selection(selection_elements=["A", "E", "I", "O", "U"],
                           val=[0.2, 0.2, 0.2, 0.2, 0.2],
                           len_query=3),
            msg="equally probable items should be in alphabetical order")

        self.assertEqual(["A", "U", "I"],
                         best_selection(
                             selection_elements=["A", "E", "I", "O", "U"],
                             val=[0.4, 0.1, 0.0, 0.1, 0.4],
                             len_query=3,
                             always_included=["I"]),
                         msg="always_included items should be in the results")

        self.assertEqual(
            ["A", "U", "E"],
            best_selection(selection_elements=["A", "E", "I", "O", "U"],
                           val=[0.4, 0.1, 0.0, 0.1, 0.4],
                           len_query=3,
                           always_included=["<"]),
            msg="Always included items are only applied if present")


class TestNBestStimuliAgent(unittest.TestCase):
    """Tests for NBestStimuliAgent"""

    def test_highest_probs(self):
        """Test the NBestStimuliAgent should select symbols with the highest
        probabilities"""
        agent = NBestStimuliAgent(alphabet=["A", "E", "I", "O", "U"],
                                  len_query=3)
        self.assertEqual(["A", "I", "U"],
                         agent.return_stimuli(
                             list_distribution=[[0.4, 0.1, 0.25, 0.1, 0.15]],
                             constants=None))

    def test_multiple_value_lists(self):
        """Test with multiple values in the list distribution."""
        agent = NBestStimuliAgent(alphabet=["A", "E", "I", "O", "U"],
                                  len_query=3)
        self.assertEqual(
            ["E", "O", "A"],
            agent.return_stimuli(
                list_distribution=[[0.4, 0.1, 0.25, 0.1, 0.15],
                                   [0.15, 0.4, 0.1, 0.25, 0.1]],
                constants=None),
            msg="Last list of probabilities should be used for selection")

    def test_all_equal_probs(self):
        """Test where all probabilities are equal"""
        agent = NBestStimuliAgent(alphabet=["A", "E", "I", "O", "U"],
                                  len_query=3)

        self.assertEqual(["A", "E", "I"],
                         agent.return_stimuli(
                             list_distribution=[[0.2, 0.2, 0.2, 0.2, 0.2]],
                             constants=None),
                         msg="Should return the first n values")

    def test_constants(self):
        """Test with constant items included in results."""
        agent = NBestStimuliAgent(alphabet=["A", "E", "I", "O", "U"],
                                  len_query=3)

        self.assertEqual(["E", "I", "U"],
                         agent.return_stimuli(
                             list_distribution=[[0.1, 0.3, 0.3, 0.2, 0.1]],
                             constants=["U"]))


class TestRandomStimuliAgent(unittest.TestCase):
    """Tests for RandomStimuliAgent"""

    def test_randomness(self):
        """Test that symbols are not selected based on probabilities."""
        agent = RandomStimuliAgent(alphabet=["A", "E", "I", "O", "U"],
                                   len_query=3)

        # Generate 100 queries
        queries = [
            agent.return_stimuli([[0.3, 0.3, 0.2, 0.1, 0.1]])
            for i in range(100)
        ]
        # Turn each list of symbols into a string to make them easier to compare
        strings = [''.join(query) for query in queries]
        self.assertTrue(len(set(strings)) > 1,
                        msg="All queries should not be the same")

        all_queries: str = ''.join(strings)
        self.assertTrue(
            "O" in all_queries or "U" in all_queries,
            msg="Not only the highest probability symbols should be returned.")

    def test_ordering(self):
        """Test that ordering is not always alphabetical"""
        agent = RandomStimuliAgent(alphabet=["A", "E", "I", "O", "U"],
                                   len_query=3)
        queries = [
            agent.return_stimuli([[0.3, 0.3, 0.2, 0.1, 0.1]])
            for i in range(10)
        ]
        self.assertFalse(all(sorted(query) == query for query in queries))

    def test_constants(self):
        """Test that constants are always added."""
        agent = RandomStimuliAgent(alphabet=["A", "E", "I", "O", "U"],
                                   len_query=3)
        query = agent.return_stimuli([[0.3, 0.3, 0.2, 0.1, 0.1]],
                                     constants=["O"])
        self.assertTrue("O" in query)


if __name__ == '__main__':
    unittest.main()
