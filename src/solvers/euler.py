from src.solvers.ode_solver import ODESolver

class EulerSolver(ODESolver):

    def step(self, func, X_t, t, dt):

        v_t = func(X_t, t)
        X_next = X_t + dt*v_t

        return X_next, {
            "X_t": X_t.detach(),
            "v_t": v_t.detach()
        }