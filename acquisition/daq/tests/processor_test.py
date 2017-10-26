from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from daq.processor import FileWriter
from mock import mock_open, patch


def test_filewriter():
    """Test FileWriter functionality"""

    data = [[i + j for j in range(3)] for i in range(3)]
    expected_csv_rows = [b'0,1,2\r\n', b'1,2,3\r\n', b'2,3,4\r\n']

    filewriter = FileWriter('foo.csv', device_name='foo-device', fs=100,
                            channels=['c1', 'c2', 'c3'])

    m = mock_open()
    with patch('daq.processor.open', m):
        with filewriter:
            m.assert_called_once_with('foo.csv', 'w')

            handle = m()
            handle.write.assert_called_with(b'timestamp,c1,c2,c3\r\n')

            for i, row in enumerate(data):
                timestamp = float(i)
                filewriter.process(row, timestamp)
                handle.write.assert_called_with(
                    str(timestamp) + "," + expected_csv_rows[i])

        m().close.assert_called_once()


def test_filewriter_builder():
    """Test FileWriter.builder"""

    data = [[i + j for j in range(3)] for i in range(3)]
    expected_csv_rows = [b'0,1,2\r\n', b'1,2,3\r\n', b'2,3,4\r\n']

    builder = FileWriter.builder('foo.csv')

    m = mock_open()
    with patch('daq.processor.open', m):
        with builder(device_name='foo-device', fs=100,
                     channels=['c1', 'c2', 'c3']) as filewriter:
            m.assert_called_once_with('foo.csv', 'w')

            handle = m()
            handle.write.assert_called_with(b'timestamp,c1,c2,c3\r\n')

            for i, row in enumerate(data):
                timestamp = float(i)
                filewriter.process(row, timestamp)
                handle.write.assert_called_with(
                    str(timestamp) + "," + expected_csv_rows[i])

        m().close.assert_called_once()
