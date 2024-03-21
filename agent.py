from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def f(self, t, x, v, w):
        """Motion model"""
        pass

    @abstractmethod
    def f_x(self, state, control):
        """Derivative wrt state"""
        pass

    @abstractmethod
    def f_u(self, state, control):
        """Derivative wrt control"""
        pass


class Unicycle(Agent):
    def __init__(self):
        super().__init__()
        pass

    def G(self, state):
        _, _, theta = state
        return np.array([
            [np.cos(theta), 0],
            [np.sin(theta), 0],
            [0, 1]
        ])

    def f(self, t, x, v, w):
        """Unicycle model"""
        u = np.array([v, w])
        return self.G(x) @ u

    def f_x(self, state, control):
        """Derivate wrt state"""
        _, _, theta = state
        v, w = control
        return np.array([
            [0, 0, -v * np.sin(theta)],
            [0, 0, v * np.cos(theta)],
            [0, 0, 0]
        ])

    def f_u(self, state, control):
        """Derivate wrt control"""
        return self.G(state)
