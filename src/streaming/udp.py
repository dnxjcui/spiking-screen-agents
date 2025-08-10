from __future__ import annotations

import socket
import struct
import threading
import time
from typing import List, Tuple

from .base import BaseEventReceiver, EventBuffer


class UDPEventReceiver(BaseEventReceiver):
    """High-performance UDP receiver for neuromorphic events.

    Packet format:
      - First 8 bytes: little-endian uint64 packet timestamp (microseconds)
      - Followed by N events, each 16 bytes, layout '<QHHb' plus 3 pad bytes:
        timestamp(uint64), x(uint16), y(uint16), polarity(int8), pad(3 bytes)
    """

    def __init__(self, port: int = 9999, buffer_size: int = 20 * 1024 * 1024) -> None:
        super().__init__()
        self.port = port
        self.buffer_size = buffer_size
        self.socket: socket.socket | None = None
        self.thread: threading.Thread | None = None
        self._last_buffer_clear = time.time()
        self._buffer_clear_interval_seconds = 5.0
        self._last_stats_time = time.time()
        self._stats_interval_seconds = 2.0
        self._packet_latencies_us: List[float] = []
        self._max_latency_samples = 100
        self._last_throughput_time = time.time()
        self._last_throughput_bytes = 0
        self.current_throughput_mbps = 0.0

    def start(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size)
            actual = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            print(f"UDP socket RCVBUF requested={self.buffer_size/(1024*1024):.1f}MB actual={actual/(1024*1024):.1f}MB")
            self.socket.bind(("127.0.0.1", self.port))
            self.socket.settimeout(0.05)

            self.running = True
            self.thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.thread.start()
            self._start_time = time.time()
            print(f"UDP receiver started on 127.0.0.1:{self.port}")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to start UDP receiver: {exc}")
            return False

    def stop(self) -> None:
        self.running = False
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        print("UDP receiver stopped")

    # Internal
    def _receive_loop(self) -> None:
        packets_in_interval = 0
        assert self.socket is not None
        while self.running:
            try:
                data, _ = self.socket.recvfrom(self.buffer_size)
                self._process_packet(data)
                self.event_buffer.stats["packets"] += 1
                self.event_buffer.stats["bytes"] += len(data)
                packets_in_interval += 1

                now = time.time()
                if now - self._last_throughput_time >= 0.1:
                    current_bytes = self.event_buffer.stats["bytes"]
                    delta = current_bytes - self._last_throughput_bytes
                    self.current_throughput_mbps = (delta * 10.0) / (1024 * 1024)
                    self._last_throughput_time = now
                    self._last_throughput_bytes = current_bytes

                if now - self._last_buffer_clear >= self._buffer_clear_interval_seconds:
                    self._clear_socket_buffer()
                    self._last_buffer_clear = now

            except socket.timeout:
                now = time.time()
                if now - self._last_buffer_clear >= self._buffer_clear_interval_seconds:
                    self._clear_socket_buffer()
                    self._last_buffer_clear = now
                continue
            except Exception as exc:  # noqa: BLE001
                if self.running:
                    print(f"UDP receive error: {exc}")
                break

    def _process_packet(self, data: bytes) -> None:
        if len(data) < 8:
            return
        packet_receive_time = time.time()
        try:
            packet_ts_us = struct.unpack("<Q", data[:8])[0]
            latency_us = (packet_receive_time * 1_000_000.0) - packet_ts_us
            if len(self._packet_latencies_us) >= self._max_latency_samples:
                self._packet_latencies_us.pop(0)
            self._packet_latencies_us.append(latency_us)

            payload = data[8:]
            event_size = 16
            n_events = len(payload) // event_size
            if n_events == 0:
                return
            events: List[dict] = []
            for i in range(n_events):
                base = i * event_size
                ts, x, y, pol = struct.unpack_from("<QHHb", payload, base)
                if 0 <= x <= 1920 and 0 <= y <= 1080:
                    events.append({
                        "timestamp": ts,
                        "x": int(x),
                        "y": int(y),
                        "polarity": int(pol),
                        "received_time": packet_receive_time,
                    })
            if events:
                self.event_buffer.add_events(events)

            now = time.time()
            if now - self._last_stats_time >= self._stats_interval_seconds:
                self._print_stats()
                self._last_stats_time = now

        except struct.error as exc:
            print(f"Packet parsing error: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"Unexpected error processing packet: {exc}")

    def _clear_socket_buffer(self) -> None:
        assert self.socket is not None
        try:
            cleared_count = 0
            cleared_bytes = 0
            self.socket.setblocking(False)
            start = time.time()
            while True:
                try:
                    data, _ = self.socket.recvfrom(self.buffer_size)
                    cleared_count += 1
                    cleared_bytes += len(data)
                    if time.time() - start > 0.05:
                        break
                except BlockingIOError:
                    break
            self.socket.setblocking(True)
            self.socket.settimeout(0.01)
            if cleared_count > 0:
                print(f"AGGRESSIVE CLEAR: Dropped {cleared_count} packets ({cleared_bytes/1024:.1f} KB)")
                self._last_stats_time = time.time()
        except Exception:
            try:
                self.socket.setblocking(True)
                self.socket.settimeout(0.01)
            except Exception:
                pass

    def _print_stats(self) -> None:
        now = time.time()
        duration = now - getattr(self, "_start_time", now)
        pkts = self.event_buffer.stats["packets"]
        evs = self.event_buffer.stats["events"]
        byt = self.event_buffer.stats["bytes"]
        if duration <= 0:
            print(f"UDP Receiver: events={evs} packets={pkts}")
            return
        avg_mbps = (byt / (1024 * 1024)) / duration
        avg_lat = (sum(self._packet_latencies_us) / len(self._packet_latencies_us)) if self._packet_latencies_us else 0.0
        max_lat = max(self._packet_latencies_us) if self._packet_latencies_us else 0.0
        print(
            f"Events/sec: {evs/duration:.0f} | Packets: {pkts} | "
            f"Throughput: {self.current_throughput_mbps:.2f} MB/s (win) | Avg: {avg_mbps:.2f} MB/s | "
            f"Latency: avg {avg_lat/1000:.1f}ms max {max_lat/1000:.1f}ms"
        )

