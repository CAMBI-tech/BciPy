import unittest

from bcipy.task.control.query import NBestStimuliAgent, RandomStimuliAgent


class TestNBestStimuliAgent(unittest.TestCase):
    """Tests for NBestStimuliAgent"""

    def test_highest_probs(self):
        """Test the NBestStimuliAgent should select symbols with the highest
        probabilities"""
        agent = NBestStimuliAgent(alphabet=["A", "E", "I", "O", "U"],
                                  len_query=3)
        stim = agent.return_stimuli(
            list_distribution=[[0.4, 0.1, 0.25, 0.1, 0.15]], constants=None)
        self.assertTrue("A" in stim)
        self.assertTrue("I" in stim)
        self.assertTrue("U" in stim)

    def test_highest_probs_ordering(self):
        """Test the NBestStimuliAgent should order the returned symbols
        by decreasing probability."""
        agent = NBestStimuliAgent(alphabet=["A", "E", "I", "O", "U"],
                                  len_query=3)
        stims = [
            agent.return_stimuli(
                list_distribution=[[0.4, 0.1, 0.25, 0.1, 0.15]],
                constants=None) for _ in range(10)
        ]

        self.assertTrue(all([stim == ['A', 'I', 'U'] for stim in stims]),
                        msg="All queries should be the same")

    def test_multiple_value_lists(self):
        """Test with multiple values in the list distribution."""
        agent = NBestStimuliAgent(alphabet=["A", "E", "I", "O", "U"],
                                  len_query=3)
        stim = agent.return_stimuli(
            list_distribution=[[0.4, 0.1, 0.25, 0.1, 0.15],
                               [0.15, 0.4, 0.1, 0.25, 0.1]],
            constants=None)
        self.assertTrue("E" in stim)
        self.assertTrue("O" in stim)
        self.assertTrue("A" in stim)

    def test_all_equal_probs(self):
        """Test where all probabilities are equal"""
        agent = NBestStimuliAgent(alphabet=["A", "E", "I", "O", "U"],
                                  len_query=3)
        stims = [
            agent.return_stimuli(list_distribution=[[0.2, 0.2, 0.2, 0.2, 0.2]],
                                 constants=None) for _ in range(10)
        ]
        stim_strings = [''.join(sorted(stim)) for stim in stims]
        self.assertTrue(len(set(stim_strings)) > 1,
                        msg="All queries should not have the same symbols")

    def test_some_equal_probs(self):
        """Test ordering where some probabilities are equal."""
        agent = NBestStimuliAgent(alphabet=["A", "E", "I", "O", "U"],
                                  len_query=3)
        stims = [
            agent.return_stimuli(list_distribution=[[0.4, 0.1, 0.2, 0.1, 0.2]],
                                 constants=None) for _ in range(10)
        ]

        unsorted_stim_strings = [''.join(stim) for stim in stims]
        self.assertTrue(len(set(unsorted_stim_strings)) > 1,
                        msg="All queries should not have the same ordering")

        sorted_stim_strings = [''.join(sorted(stim)) for stim in stims]
        self.assertEqual(1,
                         len(set(sorted_stim_strings)),
                         msg="All queries should have the same symbols")
        self.assertEqual("AIU", list(set(sorted_stim_strings))[0])

    def test_constants(self):
        """Test with constant items included in results."""
        agent = NBestStimuliAgent(alphabet=["A", "E", "I", "O", "U"],
                                  len_query=3)

        stims = agent.return_stimuli(
            list_distribution=[[0.1, 0.3, 0.3, 0.2, 0.1]], constants=["U"])
        self.assertTrue("E" in stims)
        self.assertTrue("I" in stims)
        self.assertTrue("U" in stims)


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
