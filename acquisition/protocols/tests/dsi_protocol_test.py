from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import protocols.dsi.dsi as dsi
import protocols.dsi.dsi_protocol as dsi_protocol


def test_protocol_init_messages():
    """Should have the channels and the sample_rate."""
    p = dsi_protocol.DsiProtocol()
    channel_msg = p.init_messages[0]
    c = dsi.packet.parse(channel_msg)
    assert c.type == 'EVENT'
    assert c.event_code == 'SENSOR_MAP'
    assert c.message == ','.join(dsi.default_channels)

    c2 = dsi.packet.parse(p.init_messages[1])
    assert c2.type == 'EVENT'
    assert c2.event_code == 'DATA_RATE'
    assert c2.message == u',300'


def test_encoder():
    """It should encode array data that can be subsequently decoded."""

    data = [float(i) for i in range(25)]
    encoder = dsi_protocol.Encoder()
    encoded = encoder.encode(data)
    c = dsi.packet.parse(encoded)
    assert c.type == 'EEG_DATA'
    assert c.sensor_data == data
