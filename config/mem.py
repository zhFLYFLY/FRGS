import torch
import signal
import sys
import time

class GPUMemoryReserver:
    def __init__(self, reserve_gb=7.0, device='cuda:0'):
        self.reserve_gb = reserve_gb
        self.device = torch.device(device)
        self.tensor = None
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self.release()
        sys.exit(0)

    def reserve(self):
        if not torch.cuda.is_available():
            return False
        bytes_per_element = 4
        total_bytes = int(self.reserve_gb * (1024 ** 3))
        num_elements = total_bytes // bytes_per_element
        try:
            self.tensor = torch.empty(num_elements, dtype=torch.float32, device=self.device)
            self.tensor.fill_(0.0)
            return True
        except RuntimeError:
            return False

    def release(self):
        if self.tensor is not None:
            del self.tensor
            torch.cuda.empty_cache()
            self.tensor = None

    def keep_alive(self):
        while self.running:
            time.sleep(1)

if __name__ == "__main__":
    reserver = GPUMemoryReserver(reserve_gb=7.0, device='cuda:0')
    if reserver.reserve():
        reserver.keep_alive()