"""Versioned, validated localhost telemetry shared by the launcher and rover."""
from __future__ import annotations

from collections import deque
import json
import math
import socket
import threading
import time

MAX_PACKET = 65507


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def decode_packet(data: bytes):
    try:
        packet = json.loads(data)
    except (ValueError, UnicodeError):
        return None
    if not isinstance(packet, dict) or packet.get('version') != 1 or not isinstance(packet.get('run_id'), str):
        return None
    kind = packet.get('type')
    if kind == 'pos':
        if not all(_finite(packet.get(k)) for k in ('x', 'y', 'speed', 'heading', 'time')):
            return None
    elif kind == 'obs':
        if not all(_finite(packet.get(k)) for k in ('x', 'y')):
            return None
    elif kind == 'path':
        points = packet.get('points')
        if not isinstance(points, list) or len(points) > 2048:
            return None
        if not all(isinstance(p, list) and len(p) == 2 and all(_finite(v) for v in p) for p in points):
            return None
    elif kind == 'status':
        if not isinstance(packet.get('state'), str) or not isinstance(packet.get('message', ''), str):
            return None
    elif kind == 'done':
        if not isinstance(packet.get('success'), bool):
            return None
    else:
        return None
    return packet


def encode_packet(packet: dict) -> bytes:
    data = json.dumps({'version': 1, **packet}, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(data) > MAX_PACKET or decode_packet(data) is None:
        raise ValueError('Invalid or oversized telemetry packet')
    return data


class UDPServer:
    """Bind before starting Webots so its initial path cannot be lost."""

    def __init__(self, run_id, host='127.0.0.1', port=0):
        self.run_id = run_id
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind((host, port))
            self.socket.settimeout(.15)
        except Exception:
            self.socket.close()
            raise
        self.port = self.socket.getsockname()[1]
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.trail = deque(maxlen=3000)
        self.obstacles = {}
        self.state = {'state': 'waiting', 'message': 'Kontrolcü bağlantısı bekleniyor',
                      'path': [], 'last_received': None, 'done': None, 'packet_count': 0}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                data, _ = self.socket.recvfrom(MAX_PACKET)
            except socket.timeout:
                continue
            except OSError:
                break
            packet = decode_packet(data)
            if packet is None or packet['run_id'] != self.run_id:
                continue
            with self.lock:
                self.state['last_received'] = time.monotonic()
                self.state['packet_count'] += 1
                kind = packet['type']
                if kind == 'pos':
                    self.state.update({k: packet[k] for k in ('x', 'y', 'speed', 'heading', 'time')})
                    self.trail.append((packet['x'], packet['y']))
                elif kind == 'path':
                    self.state['path'] = packet['points']
                elif kind == 'obs':
                    key = (round(packet['x'] * 2), round(packet['y'] * 2))
                    self.obstacles[key] = (packet['x'], packet['y'])
                    if len(self.obstacles) > 3000:
                        self.obstacles.pop(next(iter(self.obstacles)))
                elif kind == 'status':
                    self.state.update(state=packet['state'], message=packet.get('message', ''))
                elif kind == 'done':
                    self.state['done'] = packet
                    self.state['state'] = 'completed' if packet['success'] else 'failed'
                    self.state['message'] = packet.get('reason', '')

    def snapshot(self):
        with self.lock:
            return {**self.state, 'trail': list(self.trail), 'obstacles': list(self.obstacles.values())}

    def close(self):
        self.stop_event.set()
        self.socket.close()
        if self.thread.is_alive():
            self.thread.join(timeout=1)
