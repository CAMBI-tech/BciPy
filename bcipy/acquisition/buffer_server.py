
import logging
import multiprocessing as mp
from buffer import Buffer

# Commands to which the server can respond.
MSG_PUT = 'put_data'
MSG_GET_ALL = 'get_all_data'
MSG_QUERY_SLICE = 'query_slice'
MSG_QUERY = 'query_data'
MSG_COUNT = 'get_count'
MSG_EXIT = 'exit'
MSG_STARTED = 'started'

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


def _loop(mailbox, channels, archive_name):
    """Main server loop. Intended to be a Process target (and private to this
    module). Accepts messages through its mailbox queue, and takes the
    appropriate action based on the command and parameters contained within the
    message.

    Parameters
    ----------
        mailbox : Queue
            Used for inter-process communication.
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
        # (sender, (command, params))
        # where sender is either None or a Queue.
        msg = mailbox.get()
        sender, body = msg
        command, params = body
        if command == MSG_EXIT:
            delete_archive = params
            buf.cleanup(delete_archive=delete_archive)
            sender.put(('exit', 'ok'))
            break
        elif command == MSG_PUT:
            record = params
            buf.append(record)
        elif command == MSG_GET_ALL:
            sender.put(buf.all())
        elif command == MSG_COUNT:
            sender.put(len(buf))
        elif command == MSG_QUERY_SLICE:
            start, end, field = params
            logging.debug("Sending query: {}".format((start, end, field)))
            sender.put(buf.query(start, end, field))
        elif command == MSG_QUERY:
            # Generic query
            filters, ordering, max_results = params
            sender.put(buf.query_data(filters, ordering, max_results))
        elif command == MSG_STARTED:
            sender.put(('started', 'ok'))
        else:
            logging.debug("Error; message not understood: {}".format(msg))


def start(channels, archive_name, async=False):
    """Starts a server Process.

    Parameters
    ----------
        channels : list of str
            list of channel names. Data records are expected to have an entry
            for each channel.
        archive_name : str
            underlying database name
        async : boolean, optional; default False
            if true, returns immediately; otherwise waits for a response
            from the newly started server.
    Returns
    -------
        Queue used to communicate with this server instance.
    """

    msg_queue = mp.Queue()
    p = mp.Process(target=_loop,
                   args=(msg_queue, channels, archive_name))
    p.start()
    if not async:
        request = (MSG_STARTED, None)
        _rpc(msg_queue, request, wait_reply=True)

    return msg_queue


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


def _rpc(mailbox, request, win=None, wait_reply=True):
    """Makes a process call and optionally awaits its reply.

    Parameters
    ----------
        mailbox : Queue
            process queue
        request : tuple
            (command, params), where command is a str and params is a tuple.
        win : Window
            window to reload in case Windows changes focus
        wait_reply : boolean, optional
            waits (blocks) until a response is provided from the server process

    Returns
    -------
        Response from the server or None.
    """
    if wait_reply:
        m = mp.Manager()
        #Refocus on the window in case Windows changes focus
        if win:
            win.winHandle.activate()

        q = m.Queue()

        mailbox.put((q, request))
        # block until we receive something
        result = q.get()
        return result
    else:
        mailbox.put((None, request))
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


def get_data(mailbox, start=None, end=None, field='_rowid_', win=None):
    """Query the buffer for a slice of data records.

    Parameters
    ----------
        mailbox : Queue
            process queue
        start : float, optional
            timestamp of data lower bound; if missing, gets all data
        end : float, optional
            timestamp of data upper bound
        win : Window
            window to pass to _rpc for reloading
    Returns
    -------
        list of data rows within the given range.
    """
    if start is None:
        request = (MSG_GET_ALL, None)
    else:
        request = (MSG_QUERY_SLICE, (start, end, field))
    if win:
        return _rpc(mailbox, request, win)

    return _rpc(mailbox, request)


def query(mailbox, filters=[], ordering=None, max_results=None):
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

    request = (MSG_QUERY, (filters, ordering, max_results))
    return _rpc(mailbox, request)


def main():
    """Test script"""
    import numpy as np
    from record import Record
    import timeit

    n = 1000
    channel_count = 25
    channels = ["ch" + str(c) for c in range(channel_count)]

    pid1 = start(channels, 'buffer1.db')
    pid2 = start(channels, 'buffer2.db')

    starttime = timeit.default_timer()
    for i in range(n):
        d = [np.random.uniform(-1000, 1000) for cc in range(channel_count)]
        if i % 2 == 0:
            append(pid1, Record(d, i, None))
        else:
            append(pid2, Record(d, i, None))

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
