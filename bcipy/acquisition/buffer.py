"""The Buffer is used by the DataAcquisitionClient internally to store data
so it can be queried again. The default buffer uses a Sqlite3 database to
store data. By default it writes to a file called buffer.db, but this can be
configured."""
import os
import sqlite3
from collections import deque
import logging
import csv

from builtins import range
from bcipy.acquisition.record import Record

log = logging.getLogger(__name__)


class Buffer():
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

    # pylint: disable=too-many-instance-attributes
    def __init__(self, channels, chunksize=10000, archive_name='buffer.db'):

        assert channels, "Buffer wasn't given any channels!"
        assert chunksize > 0, "Chunksize for Buffer must be greater than 0!"
        assert archive_name, "An empty archive name will result in " \
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
        # pylint: disable=fixme
        # TODO: Field type should be configurable. For example, if a marker
        # is written to the buffer, this should be a string type.
        defs = ','.join([field + ' real' for field in fields])
        cursor.execute('DROP TABLE IF EXISTS data')

        log.debug(defs)
        cursor.execute('CREATE TABLE data (%s)' % defs)
        self._conn.commit()

        # Create SQL INSERT statement
        field_names = ','.join(fields)
        placeholders = ','.join('?' for i in range(len(fields)))
        self._insert_stmt = (f'INSERT INTO data ({field_names}) '
                             f'VALUES ({placeholders})')

    def _new_connection(self):
        """Returns a new database connection."""
        if self._archive_deleted:
            raise Exception("Queries are not allowed after buffer cleanup. "
                            "Database file has been deleted.")
        return sqlite3.connect(self._archive_name)

    def close(self):
        """Close the buffer."""
        self._flush()

    def cleanup(self, delete_archive=True):
        """Close connection and optionally deletes the archive database.
        Data may not be queryable after this method is called."""
        self._flush()
        self._conn.close()
        try:
            if delete_archive:
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

    def query(self, start, end=None, field="_rowid_"):
        """Query the buffer by for a slice of data.

        Parameters
        ----------
            start : number, starting value
            end : number, optional; ending value
                if missing returns the rest of the data
            field : str, optional; query field.
        Returns
        -------
            list of Records
        """
        filters = [(field, ">=", start)]
        if end:
            filters.append((field, "<", end))
        return self.query_data(filters=filters)

    def _validate_query_filters(self, filters):
        """Validates query filters."""
        # Guard against sql injection by limiting query inputs.
        valid_ops = ["<", "<=", ">", ">=", "=", "==", "!=", "<>", "IS",
                     "IS NOT", "IN"]
        valid_fields = ["_rowid_"] + self.fields

        if filters:
            for filter_field, filter_op, _value in filters:
                if filter_field not in valid_fields:
                    raise Exception("Invalid SQL data filter field. Must be "
                                    "one of: " + str(valid_fields))
                elif filter_op not in valid_ops:
                    raise Exception("Invalid SQL operator")

    def _validate_query_ordering(self, ordering):
        """Validates column ordering in sql query
        Parameters:
        -----------
            ordering: tuple(fieldname, direction)
        """

        valid_fields = ["_rowid_"] + self.fields
        valid_directions = ["asc", "desc"]
        if ordering:
            order_field, order_direction = ordering
            if order_field not in valid_fields:
                raise Exception("Invalid field for SQL order by.")
            elif order_direction not in valid_directions:
                raise Exception("Invalid order by direction.")

    def _build_query(self, filters, ordering, max_results):
        """Builds the SQL query with the provided filters.

        Parameters:
        -----------
            filters: list(tuple(field, operator, value)), optional
                list of tuples of the field_name, sql operator, and value.
            ordering: tuple(fieldname, direction), optional
                optional tuple indicating sort order
            max_results: int, optional

        Returns
        -------
            tuple(query:str, params:tuple)
        """
        select = "select _rowid_, * from data"

        where = ''
        query_params = ()
        if filters:
            self._validate_query_filters(filters)
            conditions = " AND ".join([" ".join([field, filter_op, "?"])
                                       for field, filter_op, _ in filters])
            query_params = tuple(filter_val for _, _, filter_val in filters)
            where = "where " + conditions

        order = ''
        if ordering:
            self._validate_query_ordering(ordering)
            order_field, order_direction = ordering
            order = ' '.join(["order by", order_field, order_direction])

        limit = "limit " + \
            str(max_results) if isinstance(max_results, int) else ''

        sql_query = ' '.join(filter(None, [select, where, order, limit]))
        return sql_query, query_params

    def query_data(self, filters=None, ordering=None, max_results=None):
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
        query, query_params = self._build_query(filters, ordering, max_results)

        self._flush()
        conn = self._new_connection()
        result = []
        log.debug('sql query: %s', query)
        log.debug('query params: %s', query_params)
        result = conn.execute(
            query, query_params) if query_params else conn.execute(query)

        # Return the data in the format it was provided
        return [_convert_row(r) for r in result]

    def dump_raw_data(self, raw_data_file_name: str, daq_type: str,
                      sample_rate: float):
        """Writes a raw_data csv file from the current database.

        Parameters:
        -----------
            db_name - path to the database
            raw_data_file_name - name of the file to be written; ex. raw_data.csv
            daq_type - metadata regarding the acquisition type; ex. 'DSI' or 'LSL'
            sample_rate - metadata for the sample rate; ex. 300.0
        """

        with open(raw_data_file_name, "w", encoding='utf-8', newline='') as raw_data_file:
            # write metadata
            raw_data_file.write(f"daq_type,{daq_type}\n")
            raw_data_file.write(f"sample_rate,{sample_rate}\n")

            # if flush is missing the previous content may be appended at the end
            raw_data_file.flush()

            self._flush()
            cursor = self._new_connection().cursor()
            cursor.execute("select * from data;")
            columns = [description[0] for description in cursor.description]

            csv_writer = csv.writer(raw_data_file, delimiter=',')
            csv_writer.writerow(columns)
            for row in cursor:
                csv_writer.writerow(row)


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
    return Record(data=list(row[2:]), timestamp=row[1], rownum=row[0])


def _main():
    import argparse
    import timeit
    from bcipy.acquisition.util import mock_data

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_records', default=100000, type=int,
                        help='number of records to insert; default is 100000')
    parser.add_argument('-s', '--chunk_size', default=10000, type=int,
                        help="default is 10000")
    parser.add_argument('-c', '--channel_count', default=25, type=int,
                        help="default is 25")

    args = parser.parse_args()
    channels = ["ch" + str(c) for c in range(args.channel_count)]

    print(
        (f"Running with {args.n_records} samples of {args.channel_count} ",
         f"channels each and chunksize {args.chunk_size}"))
    buf = Buffer(channels=channels, chunksize=args.chunk_size)

    starttime = timeit.default_timer()
    for record_data in mock_data(args.n_records, args.channel_count):
        timestamp = timeit.default_timer()
        buf.append(Record(record_data, timestamp, None))

    endtime = timeit.default_timer()
    totaltime = endtime - starttime

    print("Total records inserted: " + str(len(buf)))
    print("Total time: " + str(totaltime))
    print("Records per second: " + str(args.n_records / totaltime))

    print("First 5 records")
    print(buf.query(start=0, end=6))

    buf.cleanup()


if __name__ == '__main__':
    _main()
