from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Tuple

import imgui
import imgui.integrations.glfw
import glfw
from OpenGL.GL import *  # noqa: F401,F403 - used by ImGui renderer

from .base import BaseVisualizer, EventSource


DOT_SIZE = 2.0
DOT_FADE_DURATION = 0.1


class ImGuiEventVisualizer(BaseVisualizer):
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.canvas_width = 800
        self.canvas_height = 600
        self.active_dots: List[Dict] = []
        self.plot_times: deque = deque(maxlen=300)
        self.plot_event_counts: deque = deque(maxlen=300)
        self.last_plot_update = time.time()
        self.plot_update_interval = 1.0 / 60.0
        self.performance_stats = {"fps": 0.0, "events_per_sec": 0.0, "active_dots": 0, "total_events": 0}
        self.last_frame_time = time.time()
        self.frame_times: deque = deque(maxlen=60)

        self.window = None
        self.impl = None

    # Lifecycle
    def init_window(self, width: int = 1400, height: int = 900) -> bool:
        if not glfw.init():
            print("Failed to initialize GLFW")
            return False
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(width, height, "Neuromorphic Event Visualizer", None, None)
        if not self.window:
            print("Failed to create GLFW window")
            glfw.terminate()
            return False
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        imgui.create_context()
        self.impl = imgui.integrations.glfw.GlfwRenderer(self.window)
        imgui.style_colors_dark()
        return True

    def shutdown(self) -> None:
        if self.impl is not None:
            self.impl.shutdown()
        glfw.terminate()

    # BaseVisualizer
    def update(self, source: EventSource) -> None:
        now = time.time()
        events = source.get_recent_events(time_window=DOT_FADE_DURATION)
        self.active_dots.clear()
        for e in events:
            age = now - e.get("received_time", now)
            if age <= DOT_FADE_DURATION:
                alpha = max(0.0, min(1.0, (DOT_FADE_DURATION - age) / DOT_FADE_DURATION))
                self.active_dots.append({"x": e["x"], "y": e["y"], "polarity": e["polarity"], "alpha": alpha})

        # stats
        self.performance_stats["active_dots"] = len(self.active_dots)
        # events_per_sec over last second (approx using current set)
        one_sec_ago = now - 1.0
        recent_1s = [e for e in events if e.get("received_time", 0.0) >= one_sec_ago]
        self.performance_stats["events_per_sec"] = len(recent_1s)
        self.performance_stats["total_events"] += len(events)

        self.frame_times.append(now)
        if len(self.frame_times) > 1:
            fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
            self.performance_stats["fps"] = fps

        if now - self.last_plot_update >= self.plot_update_interval:
            self.plot_times.append(now)
            self.plot_event_counts.append(self.performance_stats["events_per_sec"])
            self.last_plot_update = now

    def render(self) -> None:
        imgui.new_frame()
        imgui.set_next_window_position(10, 10)
        imgui.set_next_window_size(1380, 880)
        expanded, opened = imgui.begin("Neuromorphic Event Visualizer", True, imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        if expanded:
            imgui.columns(2, "main_columns")
            imgui.set_column_width(0, 850)
            imgui.text("Live Neuromorphic Screen Capture")
            imgui.separator()
            self._render_canvas()
            imgui.next_column()
            imgui.text("Controls & Statistics")
            imgui.separator()
            self._render_stats()
            self._render_simple_plot()
        imgui.end()
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        imgui.render()
        assert self.impl is not None
        self.impl.render(imgui.get_draw_data())
        assert self.window is not None
        glfw.swap_buffers(self.window)
        if not opened:
            glfw.set_window_should_close(self.window, True)

    # Internals
    def _screen_to_canvas(self, sx: float, sy: float) -> Tuple[float, float]:
        if self.screen_width > 0 and self.screen_height > 0:
            scale_x = float(self.canvas_width) / float(self.screen_width)
            scale_y = float(self.canvas_height) / float(self.screen_height)
            return sx * scale_x, sy * scale_y
        return sx, sy

    def _render_canvas(self) -> None:
        window_size = imgui.get_window_size()
        canvas_pos = imgui.get_cursor_screen_pos()
        controls_height = 200
        canvas_size = (window_size.x - 100, window_size.y - canvas_pos.y - controls_height)
        if canvas_size[0] < 400.0:
            canvas_size = (400.0, canvas_size[1])
        if canvas_size[1] < 300.0:
            canvas_size = (canvas_size[0], 300.0)
        self.canvas_width = int(canvas_size[0])
        self.canvas_height = int(canvas_size[1])
        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect_filled(
            canvas_pos[0], canvas_pos[1], canvas_pos[0] + canvas_size[0], canvas_pos[1] + canvas_size[1],
            imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 1.0)
        )
        draw_list.add_rect(
            canvas_pos[0], canvas_pos[1], canvas_pos[0] + canvas_size[0], canvas_pos[1] + canvas_size[1],
            imgui.get_color_u32_rgba(0.4, 0.4, 0.4, 1.0)
        )
        for dot in self.active_dots:
            cx, cy = self._screen_to_canvas(dot["x"], dot["y"])
            sx = canvas_pos[0] + cx
            sy = canvas_pos[1] + cy
            if dot["polarity"] > 0:
                color = imgui.get_color_u32_rgba(0, int(255 * dot["alpha"]), 0, 255)
            else:
                color = imgui.get_color_u32_rgba(int(255 * dot["alpha"]), 0, 0, 255)
            draw_list.add_circle_filled(sx, sy, DOT_SIZE, color)
        imgui.dummy(canvas_size[0], canvas_size[1])

    def _render_stats(self) -> None:
        if imgui.collapsing_header("Statistics"):
            imgui.text(f"FPS: {self.performance_stats['fps']:.1f}")
            imgui.text(f"Events/sec: {self.performance_stats['events_per_sec']:.0f}")
            imgui.text(f"Active dots: {self.performance_stats['active_dots']}")

    def _render_simple_plot(self) -> None:
        if imgui.collapsing_header("Events Per Second (5s window)"):
            if len(self.plot_times) > 1 and len(self.plot_event_counts) > 1:
                draw_list = imgui.get_window_draw_list()
                canvas_pos = imgui.get_cursor_screen_pos()
                plot_width, plot_height = 400, 150
                draw_list.add_rect_filled(
                    canvas_pos[0], canvas_pos[1], canvas_pos[0] + plot_width, canvas_pos[1] + plot_height,
                    imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 1.0)
                )
                draw_list.add_rect(
                    canvas_pos[0], canvas_pos[1], canvas_pos[0] + plot_width, canvas_pos[1] + plot_height,
                    imgui.get_color_u32_rgba(0.4, 0.4, 0.4, 1.0)
                )
                start_time = self.plot_times[0]
                rel_times = [t - start_time for t in self.plot_times]
                time_range = max(rel_times) - min(rel_times) if len(rel_times) > 1 else 5.0
                count_range = max(self.plot_event_counts) - min(self.plot_event_counts) if len(self.plot_event_counts) > 1 else 1000.0
                time_range = 5.0 if time_range == 0 else time_range
                count_range = 1000.0 if count_range == 0 else count_range
                pts = []
                for i in range(min(len(rel_times), len(self.plot_event_counts))):
                    x = canvas_pos[0] + (rel_times[i] / time_range) * plot_width
                    y = canvas_pos[1] + plot_height - (self.plot_event_counts[i] / count_range) * plot_height
                    x = max(canvas_pos[0], min(canvas_pos[0] + plot_width, x))
                    y = max(canvas_pos[1], min(canvas_pos[1] + plot_height, y))
                    pts.append((x, y))
                for i in range(len(pts) - 1):
                    draw_list.add_line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], imgui.get_color_u32_rgba(0, 1, 0, 1.0), 2.0)
                if pts:
                    last = pts[-1]
                    draw_list.add_circle_filled(last[0], last[1], 3.0, imgui.get_color_u32_rgba(1, 1, 0, 1.0))
                imgui.dummy(plot_width, plot_height)
                imgui.text(f"Current: {self.performance_stats['events_per_sec']:.0f} events/sec")

