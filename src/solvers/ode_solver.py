from abc import ABC, abstractmethod
import torch

class ODESolver(ABC):

    def __init__(
        self,
        num_steps,
        track_history=False
    ):
        self.num_steps = num_steps
        self.track_history = track_history


    @abstractmethod
    def step(self, func, X_t, t, dt):
        pass

    def solve(self, func, X_t, t_0=0, t_1=1):

        times = torch.linspace(
            t_0,
            t_1,
            self.num_steps + 1,
            device=X_t.device
        )

        snapshot_times = [0.0, 0.5, 1.0]
        snapshots = {int(t * (self.num_steps-1)) for t in snapshot_times}
        his = []

        for i in range(self.num_steps):

            t = torch.full((X_t.shape[0], 1), times[i], device=X_t.device)
            dt = torch.full_like(t, times[i + 1] - times[i])

            X_t, info = self.step(func, X_t, t, dt)

            # track only pre-defined snapshots!
            if self.track_history and i in snapshots:
                his.append(info)

        return X_t, his


            