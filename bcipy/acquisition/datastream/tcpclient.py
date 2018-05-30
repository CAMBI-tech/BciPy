"""Test client for the TCP server."""

import logging
import socket

from bcipy.acquisition.protocols import dsi

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


class Signal:
    running = True
    count = 0


def tcp_client(host, port, signal):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))

    # Most protocols (DSI and BrainVision) do not require the client to send
    # any messages

    response = receive_packet(client)
    while signal.running and len(response) > 0:
        if response.type == 'EVENT' and response.event_code == 'SENSOR_MAP':
            logging.debug(response.message)
        elif response.type == 'EVENT' and response.event_code == 'DATA_RATE':
            logging.debug(response.message)
        elif response.type == 'EEG_DATA':
            signal.count += 1
            data = [i for i in response.sensor_data]
            print(data)
        else:
            logging.debug(response)
        response = receive_packet(client)

    client.close()
    logging.debug("Total records: %d" % (signal.count))


def receive_packet(socket, header_len=12):
    """Reads the header to get the payload length, then reads the payload."""

    header_buf = receive(socket, header_len)
    header = dsi.header.parse(header_buf)
    payload_buf = receive(socket, header.payload_length)
    return dsi.packet.parse(header_buf + payload_buf)


def receive(socket, msglen, chunksize=2048):
    """Receive an entire message from a socket, which may be chunked."""

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


if __name__ == '__main__':
    """Run with: python -m daq.datastream.tcpclient"""

    host = '127.0.0.1'
    port = 8844

    signal = Signal()
    try:
        tcp_client(host, port, signal)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        print("Total records: %d" % (signal.count))
        signal.running = False
