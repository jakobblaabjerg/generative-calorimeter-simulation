from src.solvers.ode_solver import ODESolver

class HeunSolver(ODESolver):

    def step(self, func, X_t, t, dt):

        k_1 = func(X_t, t)
        k_2 = func(X_t+dt*k_1, t+dt)

        X_next = X_t + dt * (k_1 + k_2)/2


        print("dt:", dt.mean().item())
        print("k1:", k_1.abs().mean().item())
        print("k2:", k_2.abs().mean().item())
        print("update:", (dt * (k_1 + k_2) / 2).abs().mean().item())


        return X_next, {
            "X_t": X_t.detach(),
            "v_t": ((k_1 + k_2)/2).detach()
        }