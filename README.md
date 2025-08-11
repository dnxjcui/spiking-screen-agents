# Spiking Screen Agents

Neuromorphic event streaming, visualization, and spiking neural networks (SNNs) for reinforcement learning. Supports both vision-based tasks (Pong with event-encoded frames) and state-based tasks (CartPole with direct state input).

## Structure

- `src/streaming/`
  - `base.py`: protocol-agnostic `EventBuffer` and `BaseEventReceiver`
  - `udp.py`: `UDPEventReceiver` implementation (8-byte ts + 16-byte events `<QHHb` + pad)
- `src/visualizers/`
  - `base.py`: `EventSource` protocol, `BaseVisualizer`
  - `imgui_visualizer.py`: ImGui visualizer that renders the event "screen"
- `src/envs/`
  - `factory.py`: Environment factory supporting "pong" and "cartpole"
  - `pong_env.py`: thin Gymnasium ALE/Pong wrapper (vision-based)
  - `cartpole_env.py`: CartPole-v1 wrapper (state-based)
- `src/app/`
  - `run_udp_visualizer.py`: Entry to run the visualizer using UDP as source
- `src/encoders/event_encoder.py`: frame -> ON/OFF event encoder
- `src/models/`: SNN architectures and model factory
  - `snn_actor_critic.py`: Vision-based SNN for image inputs (Pong)
  - `snn_state_policy.py`: State-based SNN for vector inputs (CartPole)
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

## Train SNNs with PPO

### CartPole (State-based SNN - Recommended for Debugging)

Train a state-based SNN on CartPole for quick debugging and validation:

```bash
python -m src.app.train_ppo --env cartpole --episodes 200 --save cartpole_snn.pt
```

### Pong (Vision-based SNN)

Train with event-encoded frames (ON/OFF from frame differences):

```bash
python -m src.app.train_ppo --env pong --episodes 50 --render human --save pong_snn.pt
```

### Training Flags:
- `--env`: environment to train on ("cartpole" or "pong")
- `--episodes`: number of training episodes
- `--render`: render mode ("human" for visual, "rgb_array" for no display)
- `--save`: path to save the trained model
- `--threshold`, `--stride`, `--lr`: encoder/training hyperparameters

## Run Inference from Live UDP Events (SNN policy)

Run the trained policy on live UDP events:

```
python -m src.app.infer_udp --model pong_snn.pt --port 9999
## Live training/testing against a windowed game via UDP events

Control a running windowed Pong-like game using UDP event stream and keyboard actions:

```
python -m src.app.live_udp_control --model pong_snn.pt --port 9999 --window-ms 50 --kill-key esc
```

Notes:
- Uses the same UDP format as the visualizer.
- Aggregates events over a 50 ms window into a 2xHxW ON/OFF tensor.
- Sends arrow key presses; press ESC to stop.

```

This constructs 2xHxW ON/OFF maps from UDP events at ~30Hz and outputs an action (noop/up/down) with probabilities.

## Evaluate Trained Models

Test a trained model visually:

```bash
python eval_model.py --model cartpole_snn.pt --episodes 5 --deterministic
python eval_model.py --model pong_snn.pt --episodes 3
```

## Notes on Extensibility

- To add new streaming protocols, implement a receiver that exposes `event_buffer.get_recent_events(window)` like `UDPEventReceiver`.
- To add new environments, create a wrapper in `src/envs/` and register it in `src/envs/factory.py`.
- For state-based environments, use the `snn_state_ac` model; for vision-based, use `snn_ac`.
- The visualizer already consumes the generic `EventSource` protocol, so it will work unchanged with new receivers.

## Architecture Notes

### SNN Models:
- **Vision-based** (`snn_ac`): Takes 2-channel event frames (ON/OFF), uses convolutional spiking layers
- **State-based** (`snn_state_ac`): Takes raw state vectors, uses fully-connected spiking layers
- Both use LIF (Leaky Integrate-and-Fire) neurons with surrogate gradients for backpropagation

### Training Improvements:
- Tracks episode rewards and individual loss components (policy, value, entropy)
- Fixed SNN state management to prevent gradient computation errors
- Support for both vision and state-based inputs in single training script

