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
    return ''.join(chunks)


def import_submodules(package, recursive=True):
    import importlib
    import pkgutil

    """ Import all submodules of a module, recursively, including subpackages.
    https://stackoverflow.com/questions/3365740/how-to-import-all-submodules

    Parameters
    ----------
        package : str | package
            name of package or package instance
        recursive : bool, optional

    Returns
    -------
        dict[str, types.ModuleType]
    """
    if type(package) == str or type(package) == unicode:
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results
