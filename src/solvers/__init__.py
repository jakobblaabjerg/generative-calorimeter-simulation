from .euler import EulerSolver
from .heun import HeunSolver

SOLVERS = {
    "euler": EulerSolver,
    "heun": HeunSolver,
}