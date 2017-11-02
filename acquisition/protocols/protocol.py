from collections import namedtuple

""""Protocols are primarily used for generating data for testing purposes"""
Protocol = namedtuple(
    'Protocol', ['encoder', 'init_messages', 'fs', 'channels'])
