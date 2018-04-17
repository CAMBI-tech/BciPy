from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import sqlite3
from collections import deque

from builtins import range
from record import Record


class Buffer(object):
    """Queryable Data Structure for caching data acquisition data.

    Parameters
    ----------
        channels: list
            list of channel names
        chunksize: int; optional
            Max number of records to keep in memory before flushing to the data
            store. There is a performance tradeoff between average insert time
            (more performant with large chunksize) and max insert time (more
             performant with small chunksize).
        archive_name: str; optional
            Name of the data store.

    http://sebastianraschka.com/Articles/2014_sqlite_in_python_tutorial.html
    """

    def __init__(self, channels, chunksize=10000, archive_name='buffer.db'):

        assert len(channels) > 0, "Buffer wasn't given any channels!"
        assert chunksize > 0, "Chunksize for Buffer must be greater than 0!"
        assert len(archive_name) > 0, "An empty archive name will result in " \
            "an in-memory database that cannot be shared across processes."

        # Flush to data store when buf is this size.
        self._chunksize = chunksize
        self._archive_name = archive_name
        self._buf = deque()
        self.start_time = None
        self._archive_deleted = False

        # There is some discussion and disagreement on re-using sqlite
        # connections between threads. If any issues are encountered, consider
        # opening a new connection for each request, which should be fairly
        # infrequent, or use a connection pool, such as the one provided by
        # the SQLAlchemy library.
        #
        # https://bugs.python.org/issue16509 (check_same_thread parameter)
        # https://stackoverflow.com/questions/14511337/efficiency-of-reopening-sqlite-database-after-each-query
        self._conn = sqlite3.connect(archive_name, check_same_thread=False)
        cursor = self._conn.cursor()

        # Create a data table with the correct number of channels (+ timestamp)
        fields = ['timestamp'] + channels
        self.fields = fields
        defs = ','.join([field + ' real' for field in fields])
        cursor.execute('DROP TABLE IF EXISTS data')
        cursor.execute('CREATE TABLE data (%s)' % defs)
        self._conn.commit()

        # Create SQL INSERT statement
        field_names = ','.join(fields)
        placeholders = ','.join('?' for i in range(len(fields)))
        self._insert_stmt = 'INSERT INTO data (%s) VALUES (%s)' % \
            (field_names, placeholders)

    @classmethod
    def with_opts(cls, options):
        """Returns a builder than constructs a new Buffer with the given
        options."""
        def build(channels):
            return Buffer(channels, **options)
        return build

    @classmethod
    def builder(cls, archive_name):
        """Returns a builder than constructs a new Buffer with the given
        options."""
        def build(channels):
            return Buffer(channels, archive_name=archive_name)
        return build

    def _new_connection(self):
        """Returns a new database connection."""
        if self._archive_deleted:
            raise Exception("Queries are not allowed after buffer cleanup. "
                            "Database file has been deleted.")
        return sqlite3.connect(self._archive_name)

    def close(self):
        self._flush()

    def cleanup(self):
        """Deletes the archive database. Data cannot be queried after this
        method is called."""

        self._conn.close()
        try:
            os.remove(self._archive_name)
            self._archive_deleted = True
        except OSError:
            pass

    def append(self, record):
        """Append the list of readings for the given timestamp.

        Parameters
        ----------
            record: Record (named tuple)
                sensor reading data and timestamp attributes
        """
        if self.start_time is None:
            self.start_time = record.timestamp

        self._buf.append(record)

        if len(self._buf) >= self._chunksize:
            self._flush()

    def _flush(self):
        """Writes data to the datastore and empties the buffer."""

        data = [_adapt_record(self._buf.popleft())
                for i in range(len(self._buf))]

        if data:
            # Performs writes in a single transaction;
            with self._conn as conn:
                conn.executemany(self._insert_stmt, data)

    def __len__(self):
        self._flush()

        cur = self._new_connection().cursor()
        cur.execute("select count(*) from data", ())
        return cur.fetchone()[0]

    def __str__(self):
        return "Buffer<archive_name={}>".format(self._archive_name)

    def all(self):
        """Returns all data in the buffer."""

        return self.query_data(filters=[("timestamp", ">=", self.start_time)])

    def latest(self, limit=1000):
        """Get the n most recent number of items.

        Parameters
        ----------
            limit : int
                maximum number of items to retrieve
        Returns
        -------
            list of tuple(timestamp, data); will be in descending order by
            timestamp.
        """

        return self.query_data(ordering=("timestamp", "desc"),
                               max_results=limit)

    def query(self, start, end=None):
        """Query the buffer for a time slice.

        Parameters
        ----------
            start : timestamp
            end : timestamp, optional
                if missing returns the rest of the data
        Returns
        -------
            list of tuple(timestamp, data)
        """
        if end:
            return self.query_data(filters=[("timestamp", ">=", start),
                                            ("timestamp", "<", end)])
        else:
            return self.query_data(filters=[("timestamp", ">=", start)])

    def query_data(self, filters=[], ordering=None, max_results=None):
        """Query the data with the provided filters.

        Parameters:
        -----------
            filters: list(tuple(field, operator, value)), optional
                list of tuples of the field_name, sql operator, and value.
            ordering: tuple(fieldname, direction), optional
                optional tuple indicating sort order
            max_results: int, optional

        Returns
        -------
            list of Records matching the query.
        """

        # Guard against sql injection by limiting query inputs.
        valid_ops = ["<", "<=", ">", ">=", "=", "==", "!=", "<>", "IS",
                     "IS NOT", "IN"]
        for field, op, value in filters:
            if not field in self.fields:
                raise Exception("Invalid SQL data filter field. Must be one "\
                 "of: " + str(self.fields))
            elif not op in valid_ops:
                raise Exception("Invalid SQL operator")

        select = "select * from data"

        where = ''
        params = ()
        if filters:
            where = "where " + " AND ".join([" ".join([field, op, "?"])
                                             for field, op, value in filters])
            params = tuple(value for field, op, value in filters)

        order = ''
        if ordering is not None:
            f, direction = ordering
            if not f in self.fields:
                raise Exception("Invalid field for SQL order by.")
            elif not direction in ["asc", "desc"]:
                raise Exception("Invalid order by direction.")

            order = ' '.join(["order by", f, direction])

        limit = "limit " + \
            str(max_results) if isinstance(max_results, int) else ''

        q = ' '.join(filter(None, [select, where, order, limit]))

        self._flush()
        conn = self._new_connection()
        result = []

        print(q)
        result = conn.execute(q, params) if params else conn.execute(q)

        # Return the data in the format it was provided
        return [_convert_row(r) for r in result]


def _adapt_record(record):
    """Adapt for insertion into database.

    Parameters
    ----------
        record : Record
    """
    return tuple([record.timestamp] + record.data)


def _convert_row(row):
    """Convert from database row to Record.

    Parameters
    ----------
        row : tuple
    """
    return Record(data=list(row[1:]), timestamp=row[0])


def _main():
    import argparse
    import timeit
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_records', default=100000, type=int,
                        help='number of records to insert; default is 100000')
    parser.add_argument('-s', '--chunk_size', default=10000, type=int,
                        help="default is 10000")
    parser.add_argument('-c', '--channel_count', default=25, type=int,
                        help="default is 25")

    args = parser.parse_args()
    n = args.n_records
    chunksize = args.chunk_size
    channel_count = args.channel_count

    channels = ["ch" + str(c) for c in range(channel_count)]

    print("Running with %d samples of %d channels each and chunksize %d" %
          (n, channel_count, chunksize))
    b = Buffer(channels=channels, chunksize=chunksize)

    def data(n, c):
        """Generater for mock data"""
        i = 0
        while i < n:
            yield [np.random.uniform(-1000, 1000) for cc in range(c)]
            i += 1

    starttime = timeit.default_timer()
    for d in data(n, channel_count):
        timestamp = timeit.default_timer()
        b.append(Record(d, timestamp))

    endtime = timeit.default_timer()
    totaltime = endtime - starttime

    print("Total records inserted: " + str(len(b)))
    print("Total time: " + str(totaltime))
    print("Records per second: " + str(n / totaltime))

    b.cleanup()


if __name__ == '__main__':
    _main()
