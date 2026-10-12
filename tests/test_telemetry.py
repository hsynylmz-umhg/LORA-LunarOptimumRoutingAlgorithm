import json
import socket
import time
import unittest

from src.telemetry import UDPServer, decode_packet, encode_packet


class TelemetryTests(unittest.TestCase):
    def test_full_large_route_keeps_goal(self):
        points = [[float(i), float(i)] for i in range(600)]
        data = encode_packet(dict(run_id='test', type='path', points=points))
        self.assertGreater(len(data), 2048)
        self.assertEqual(decode_packet(data)['points'][-1], [599., 599.])
        self.assertIsNone(decode_packet(data[:-5]))

    def test_malformed_and_nonfinite(self):
        for data in (b'bad', b'[]', b'{}', b'\xff', b'{"version":1,"run_id":"x","type":"pos","x":NaN}'):
            self.assertIsNone(decode_packet(data))
        with self.assertRaises(ValueError):
            encode_packet(dict(type='path', run_id='x', points=[[float('inf'), 0.]]))

    def test_live_loopback_and_session_filter(self):
        server = UDPServer('current')
        server.start()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                for run_id in ('old', 'current'):
                    sender.sendto(encode_packet(dict(run_id=run_id, type='status', state='ready')), ('127.0.0.1', server.port))
            deadline = time.monotonic() + 2
            while server.snapshot()['packet_count'] < 1 and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(server.snapshot()['packet_count'], 1)
            self.assertEqual(server.snapshot()['state'], 'ready')
        finally:
            server.close()
