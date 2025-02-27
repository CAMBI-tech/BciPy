import unittest
from unittest.mock import Mock

from bcipy.simulator.data.data_engine import QueryFilter
from bcipy.simulator.data.sampler.replay_sampler import ReplaySampler
from bcipy.simulator.data.trial import Trial
from bcipy.simulator.exceptions import IncompatibleData
from bcipy.simulator.util.state import SimState


class ReplaySamplerTest(unittest.TestCase):
    """Tests for Replay sampler class."""

    def test_init(self):
        """Test initialization"""
        data_source = Mock()
        data_source.source_dirs = []
        self.assertRaises(IncompatibleData, lambda: ReplaySampler(data_source))

        data_source.source_dirs = ['source1']
        sampler = ReplaySampler(data_source)
        self.assertEqual('source1', sampler.current_source)

    def test_filters(self):
        """Test query filters"""
        data_source = Mock()
        data_source.source_dirs = ['source1']
        sampler = ReplaySampler(data_source)

        state = SimState(target_symbol='H',
                         current_sentence='',
                         target_sentence='HELLO',
                         display_alphabet=['A', 'B', 'C'],
                         inquiry_n=0,
                         series_n=1)

        expected = [
            QueryFilter('source', '==', 'source1'),
            QueryFilter('series', '==', 1),
            QueryFilter('series_inquiry', '==', 0),
            QueryFilter('symbol', '==', 'A')
        ]
        self.assertEqual(expected, sampler.query_filters(state, 'A'))

    def test_sample(self):
        """Test sampling"""
        data_source = Mock()
        data_source.source_dirs = ['source1']
        data_source.query.return_value = [
            Trial(source='source1',
                  series=1,
                  series_inquiry=0,
                  inquiry_n=0,
                  inquiry_pos=0,
                  symbol='A',
                  target=0,
                  eeg=Mock())
        ]

        sampler = ReplaySampler(data_source)

        state = SimState(target_symbol='H',
                         current_sentence='',
                         target_sentence='HELLO',
                         display_alphabet=['A', 'B', 'C'],
                         inquiry_n=0,
                         series_n=1)
        sampler.sample(state)
        self.assertEqual(
            3,
            data_source.query.call_count,
            msg="should have been called for each displayed symbol")

    def test_next_source(self):
        """Test incrementing the source"""
        data_source = Mock()

        data_source.source_dirs = ['source1', 'source2']
        sampler = ReplaySampler(data_source)
        self.assertEqual('source1', sampler.current_source)

        sampler.next_source()
        self.assertEqual('source2', sampler.current_source)
