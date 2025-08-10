import time
import numpy as np
import torch
from src.utils import Config
from src.all_code import SNNPolicy

# ---------------------------
# UDP Inference (optional)
# ---------------------------

class UDPEventInference:
    """Listen for 16-byte DVSEvents sent after an 8-byte packet timestamp.
    Build ON/OFF images and query the trained SNN at a fixed interval."""
    def __init__(self, cfg: Config, port: int = 9999):
        import socket, struct
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 20 * 1024 * 1024)
        self.socket.bind(("127.0.0.1", port))
        self.socket.settimeout(0.05)
        self.struct = struct
        self.cfg = cfg
        self.encoder_w = cfg.width
        self.encoder_h = cfg.height
        self.event_img = np.zeros((2, self.encoder_h, self.encoder_w), dtype=np.float32)
        self.last_infer = time.time()
        self.infer_interval = 0.033  # ~30Hz

    def _ingest_packet(self, data: bytes):
        if len(data) < 8:
            return 0
        # first 8 bytes are packet timestamp (uint64, little endian)
        # remaining are a sequence of 16-byte events: <QHHb + 3 pad>
        evt_size = 16
        payload = data[8:]
        n = len(payload) // evt_size
        for i in range(n):
            base = i * evt_size
            # unpack first 13 bytes: timestamp (Q), x (H), y (H), polarity (b)
            ts, x, y, pol = self.struct.unpack_from("<QHHb", payload, base)
            # scale to encoder size (assuming source is 1920x1080)
            sx, sy = 1920, 1080
            xx = int((x / sx) * self.encoder_w)
            yy = int((y / sy) * self.encoder_h)
            if 0 <= xx < self.encoder_w and 0 <= yy < self.encoder_h:
                if pol > 0:
                    self.event_img[0, yy, xx] = 1.0
                else:
                    self.event_img[1, yy, xx] = 1.0
        return n

    def run(self, model_path: str):
        # Load policy
        pol = SNNPolicy(in_ch=2, n_actions=3, beta=self.cfg.beta, device=self.cfg.device)
        state = torch.load(model_path, map_location=self.cfg.device)
        pol.load_state_dict(state["model_state"])
        pol.eval()
        pol.reset_state()

        print("Listening on UDP 127.0.0.1:9999 ... Press Ctrl+C to stop.")
        try:
            while True:
                try:
                    data, _ = self.socket.recvfrom(64 * 1024 * 1024)
                    ne = self._ingest_packet(data)
                except Exception:
                    ne = 0

                now = time.time()
                if now - self.last_infer >= self.infer_interval:
                    x = torch.from_numpy(self.event_img).unsqueeze(0)  # 1x2xHxW
                    with torch.no_grad():
                        logits = pol(x)
                        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                        act = int(np.argmax(probs))
                        # 0=noop, 1=up, 2=down
                        print(f"prob(noop,up,down)={np.round(probs,4)} -> action={act}")
                    # decay the event image
                    self.event_img *= 0.1
                    self.last_infer = now
        except KeyboardInterrupt:
            print("Stopping UDP inference.")
        finally:
            self.socket.close()