#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
import argparse

import glfw

from src.streaming.udp import UDPEventReceiver
from src.visualizers.imgui_visualizer import ImGuiEventVisualizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ImGui visualizer for UDP neuromorphic events")
    parser.add_argument("--port", type=int, default=9999, help="UDP port to listen on")
    parser.add_argument("--width", type=int, default=1920, help="Screen width for coordinate scaling")
    parser.add_argument("--height", type=int, default=1080, help="Screen height for coordinate scaling")
    args = parser.parse_args()

    receiver = UDPEventReceiver(port=args.port, buffer_size=20 * 1024 * 1024)
    if not receiver.start():
        return 1

    vis = ImGuiEventVisualizer(screen_width=args.width, screen_height=args.height)
    if not vis.init_window():
        receiver.stop()
        return 1

    try:
        while not glfw.window_should_close(vis.window):  # type: ignore[arg-type]
            glfw.poll_events()
            assert vis.impl is not None
            vis.impl.process_inputs()
            vis.update(receiver.event_buffer)
            vis.render()
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        vis.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

