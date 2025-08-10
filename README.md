# Spiking Screen Agents

Neuromorphic event streaming, visualization, and a simple spiking policy (SNN) that learns to play Pong using event-encoded frames.

## Structure

- `src/streaming/`
  - `base.py`: protocol-agnostic `EventBuffer` and `BaseEventReceiver`
  - `udp.py`: `UDPEventReceiver` implementation (8-byte ts + 16-byte events `<QHHb` + pad)
- `src/visualizers/`
  - `base.py`: `EventSource` protocol, `BaseVisualizer`
  - `imgui_visualizer.py`: ImGui visualizer that renders the event "screen"
- `src/envs/`
  - `factory.py`: `make_env("pong")` returns a Pong wrapper and action mapper
  - `pong_env.py`: thin Gymnasium ALE/Pong wrapper
- `src/app/`
  - `run_udp_visualizer.py`: Entry to run the visualizer using UDP as source
- `src/all_code.py`: SNN model, encoder, training loop, and CLI
- `src/event_receiver.py`: UDP inference helper (policy on live event stream)

## Dependencies

Python 3.10+ recommended.

Required packages:

```
pip install gymnasium[atari,accept-rom-license] snntorch torch numpy opencv-python
pip install PyOpenGL glfw imgui[glfw]
```

Notes:
- On Windows, ensure proper GPU drivers for torch if you want CUDA. CPU also works.
- Atari ROMs are handled by `gymnasium[atari,accept-rom-license]` extras.

## UDP Packet Format

- First 8 bytes: little-endian uint64 packet timestamp (microseconds)
- Repeated 16-byte events: `<QHHb` + 3 pad bytes
  - `timestamp` (uint64), `x` (uint16), `y` (uint16), `polarity` (int8), `pad` (3 bytes)
- Coordinates expected up to 1920x1080; visualizer scales to canvas.

## Run the Visualizer (UDP)

Start the visualizer:

```
python -m src.app.run_udp_visualizer --port 9999 --width 1920 --height 1080
```

Send properly formatted UDP packets to `127.0.0.1:9999`. The canvas will show green dots for ON and red dots for OFF events, fading over 100ms.

## Train the SNN on Pong

Train with event-encoded frames (ON/OFF from frame diffs):

```
python -m src.all_code train --episodes 50 --render --save pong_snn.pt
```

Flags:
- `--episodes`: number of training episodes
- `--render`: show Gym's RGB render (optional)
- `--save`: path to save the trained model
- `--threshold`, `--stride`, `--lr`: encoder/training knobs

## Run Inference from Live UDP Events (SNN policy)

Run the trained policy on live UDP events:

```
python -m src.all_code infer-udp --model pong_snn.pt --port 9999
```

This constructs 2xHxW ON/OFF maps from UDP events at ~30Hz and outputs an action (noop/up/down) with probabilities.

## Notes on Extensibility

- To add new streaming protocols, implement a receiver that exposes `event_buffer.get_recent_events(window)` like `UDPEventReceiver`.
- To add new envs (e.g., Pacman), add a new wrapper and register it in `src/envs/factory.py`.
- The visualizer already consumes the generic `EventSource` protocol, so it will work unchanged with new receivers.

