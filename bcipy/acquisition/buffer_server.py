"""Module for creating server (Process) that manages a data buffer."""
import logging
import multiprocessing as mp
from bcipy.acquisition.buffer import Buffer

# Commands to which the server can respond.
MSG_PUT = 'put_data'
MSG_GET_ALL = 'get_all_data'
MSG_QUERY_SLICE = 'query_slice'
MSG_QUERY = 'query_data'
MSG_COUNT = 'get_count'
MSG_EXIT = 'exit'
MSG_STARTED = 'started'

log = logging.getLogger(__name__)


def _loop(msg_queue, response_queue, channels, archive_name):
    """Main server loop. Intended to be a Process target (and private to this
    module). Accepts messages through its mailbox queue, and takes the
    appropriate action based on the command and parameters contained within the
    message.

    Parameters
    ----------
        msq_queue : Queue
            Used for receiving inter-process communication.
        response_queue : Queue
            Used for pushing responses
        channels : list of str
            list of channel names in the underlying data table. Any records
            written to the buffer are expected to have an entry for each
            channel.
        archive_name : str
            sqlite database name
    """
    buf = Buffer(channels=channels, archive_name=archive_name)

    while True:
        # Messages should be tuples with the structure:
        # (command, params)
        msg = msg_queue.get()
        command, params = msg
        if command == MSG_EXIT:
            buf.cleanup(delete_archive=params)
            response_queue.put(('exit', 'ok'))
            break
        elif command == MSG_PUT:
            # params is the record to put
            buf.append(params)
        elif command == MSG_GET_ALL:
            response_queue.put(buf.all())
        elif command == MSG_COUNT:
            response_queue.put(len(buf))
        elif command == MSG_QUERY_SLICE:
            row_start, row_end, field = params
            log.debug("Sending query: %s", (row_start, row_end, field))
            response_queue.put(buf.query(row_start, row_end, field))
        elif command == MSG_QUERY:
            # Generic query
            filters, ordering, max_results = params
            response_queue.put(buf.query_data(filters, ordering, max_results))
        elif command == MSG_STARTED:
            response_queue.put(('started', 'ok'))
        else:
            log.debug("Error; message not understood: %s", msg)


def new_mailbox():
    """Creates a new mailbox used to communicate with a buffer process, but 
    does not create or start the process.

    Returns
    -------
        Tuple of Queues used to communicate with this server instance.
    """
    msg_queue = mp.Queue()
    response_queue = mp.Queue()
    return (msg_queue, response_queue)


def start_server(mailbox, channels, archive_name):
    """Starts a server process using the provided mailbox for communication.
    
    Parameters
    ----------
        mailbox: tuple of Queues used to communicate with this server instance.
        channels : list of str
            list of channel names. Data records are expected to have an entry
            for each channel.
        archive_name : str
            underlying database name
    """
    log.debug("Starting the database server")
    msg_queue, response_queue = mailbox
    server_process = mp.Process(target=_loop,
                                args=(msg_queue, response_queue, channels,
                                      archive_name))
    server_process.start()

    request = (MSG_STARTED, None)
    return _rpc(mailbox, request, wait_reply=True)


def start(channels, archive_name, asynchronous=False):
    """Starts a server Process.

    Parameters
    ----------
        channels : list of str
            list of channel names. Data records are expected to have an entry
            for each channel.
        archive_name : str
            underlying database name
        asynchronous : boolean, optional; default False
            if true, returns immediately; otherwise waits for a response
            from the newly started server.
    Returns
    -------
        Tuple of Queues used to communicate with this server instance.
    """
    msg_queue = mp.Queue()
    response_queue = mp.Queue()
    mailbox = (msg_queue, response_queue)
    server_process = mp.Process(target=_loop, args=(
        msg_queue, response_queue, channels, archive_name))
    server_process.start()
    if not asynchronous:
        request = (MSG_STARTED, None)
        _rpc(mailbox, request, wait_reply=True)

    return mailbox

def stop(mailbox, delete_archive=True):
    """Stops the process associated with the provided mailbox.

    Parameters
    ----------
        mailbox : Queue
            queue used to communicate with the process.
        delete_archive : boolean, optional
            if true, deletes the archive on exit
    """
    request = (MSG_EXIT, delete_archive)
    return _rpc(mailbox, request)


def _rpc(mailbox, request, wait_reply=True):
    """Makes a process call and optionally awaits its reply.

    Parameters
    ----------
        mailbox : Queue
            process queue
        request : tuple
            (command, params), where command is a str and params is a tuple.
        wait_reply : boolean, optional
            waits (blocks) until a response is provided from the server process

    Returns
    -------
        Response from the server or None.
    """
    msg_queue, response_queue = mailbox
    if wait_reply:
        msg_queue.put(request)
        # block until we receive something
        result = response_queue.get()
        return result

    msg_queue.put(request)
    return None


def append(mailbox, record):
    """Write the record to the given buffer process. Returns immediately and
     does not wait for a response.

    Parameters
    ----------
        mailbox : Queue
            process queue
        record : Record
            data row to write.
    """
    request = (MSG_PUT, record)
    _rpc(mailbox, request, wait_reply=False)


def count(mailbox):
    """Get the count of the total number of entries in the buffer.

    Parameters
    ----------
        mailbox : Queue
            process queue
    Returns
    -------
        int of the buffer length
    """
    request = (MSG_COUNT, None)
    return _rpc(mailbox, request)

# pylint: disable=redefined-outer-name


def get_data(mailbox, start=None, end=None, field='_rowid_'):
    """Query the buffer for a slice of data records.

    Parameters
    ----------
        mailbox : Queue
            process queue
        start : float, optional
            timestamp of data lower bound; if missing, gets all data
        end : float, optional
            timestamp of data upper bound
    Returns
    -------
        list of data rows within the given range.
    """
    if start is None:
        request = (MSG_GET_ALL, None)
    else:
        request = (MSG_QUERY_SLICE, (start, end, field))

    return _rpc(mailbox, request)


def query(mailbox, filters=None, ordering=None, max_results=None):
    """Query the buffer for data.

    Parameters:
    -----------
        filters: list(tuple(field, operator, value)), optional
            list of tuples of the field_name, sql operator, and value.
        ordering: tuple(fieldname, direction), optional
            optional tuple indicating sort order
        max_results: int, optional

    Returns
    -------
        list of data rows meeting the query criteria.
    """
    filters = filters or []
    request = (MSG_QUERY, (filters, ordering, max_results))
    return _rpc(mailbox, request)


def main():
    """Test script"""
    import numpy as np
    from bcipy.acquisition.record import Record
    import timeit

    n_rows = 1000
    channel_count = 25
    channels = ["ch" + str(c) for c in range(channel_count)]

    pid1 = start(channels, 'buffer1.db')
    pid2 = start(channels, 'buffer2.db')

    starttime = timeit.default_timer()
    for i in range(n_rows):
        data = [np.random.uniform(-1000, 1000) for _ in range(channel_count)]
        if i % 2 == 0:
            append(pid1, Record(data, i, None))
        else:
            append(pid2, Record(data, i, None))

    endtime = timeit.default_timer()
    totaltime = endtime - starttime

    print("Records inserted in buffer 1: {}".format(count(pid1)))
    print("Records inserted in buffer 2: {}".format(count(pid2)))

    print("Total insert time: " + str(totaltime))

    query_n = 5
    data = get_data(pid1, 0, query_n)
    print("Sample records from buffer 1 (query < {}): {}".format(query_n,
                                                                 data))
    stop(pid1)
    stop(pid2)


if __name__ == '__main__':
    import sys
    if sys.version_info >= (3, 0, 0):
        # Only available in Python 3; allows us to test process code as it
        # behaves in Windows environments.
        mp.set_start_method('spawn')
    main()
