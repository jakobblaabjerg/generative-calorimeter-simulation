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

    def solve(self, func, noise):

        history = [] # for tracking velocities 
        snapshots = {int(t * (self.num_steps-1)) for t in [0.0, 0.5, 1.0]}
        dt_scalar = 1 / self.num_steps
        X_t = noise.clone()

        for i in range(self.num_steps):

            t = torch.full((X_t.shape[0], 1), i*dt_scalar, device=X_t.device)
            dt = torch.full_like(t, dt_scalar)
            X_t, info = self.step(func, X_t, t, dt)

            # track only pre-defined snapshots!
            if self.track_history and i in snapshots:
                history.append(info)

        return X_t, history