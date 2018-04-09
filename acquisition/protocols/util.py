"""Utility functions for protocols."""
from __future__ import absolute_import, division, print_function

def receive(socket, msglen, chunksize=2048):
    """Receive an entire message from a socket, which may be chunked.

    Parameters
    ----------
        socket : socket.socket
            object from which to receive data
        msglen : int
            length of the entire message
        chunksize : int ; optional
            messages will be received in chunksize lengths and joined together.

    Return
    ------
        str - entire message.
    """

    chunks = []
    bytes_received = 0
    while bytes_received < msglen:
        recv_len = min(msglen - bytes_received, chunksize)
        chunk = socket.recv(recv_len)
        if chunk == '':
            raise RuntimeError("socket connection broken")
        chunks.append(chunk)
        bytes_received = bytes_received + len(chunk)
    return b''.join(chunks)
