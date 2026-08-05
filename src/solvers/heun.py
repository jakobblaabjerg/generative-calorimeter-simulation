from src.solvers.ode_solver import ODESolver

class HeunSolver(ODESolver):

    def step(self, func, X_t, t, dt):

        k_1 = func(X_t, t)
        k_2 = func(X_t+dt*k_1, t+dt)

        X_next = X_t + dt * (k_1 + k_2)/2

        return X_next, {
            "X_t": X_t.detach(),
            "v_t": ((k_1 + k_2)/2).detach()
        }