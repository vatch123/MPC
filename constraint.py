from abc import ABC, abstractmethod
from agent import Agent
import numpy as np

class Constraint:
    """Abstract class for constraints."""
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def h(self, state):
        pass
    
    @abstractmethod
    def h_x(self, state):
        pass
    
    @abstractmethod
    def h_xx(self, state):
        pass
    
    @abstractmethod
    def h_u(self, state, control):
        pass

class CBFConstraint:
    """Convert a constraint to a CBF constraint."""
    def __init__(
      self,
      constraint: Constraint,
      agent: Agent,
      alpha,
      epsilon
    ) -> None:
        self.constraint = constraint
        self.agent = agent
        self.alpha = alpha
        self.epsilon = epsilon

        self.f = self.agent.f
        self.f_x = self.agent.f_x
        self.f_u = self.agent.f_u

        self.h = self.constraint.h
        self.h_x = self.constraint.h_x
        self.h_xx = self.constraint.h_xx
        self.h_u = self.constraint.h_u
  
    def g(self, state, control):
        v, w = control
        return - self.h_x(state) @ self.f(0, state, v, w) - self.alpha * self.h(state)
    
    def g_x(self, state, control):
        v, w = control
        return - self.h_xx(state) @ self.f(0, state, v, w) - self.h_x(state) @ self.f_x(state, control) - self.alpha * self.h_x(state)

    def g_u(self, state, control):
        return - self.h_x(state) @ self.f_u(state, control)


class CircularConstraint(Constraint):
    """Circular constraint."""
    def __init__(self, radius, x_coord, y_coord) -> None:
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.radius = radius
        self.obstacle = np.array([x_coord, y_coord, 0])
        self.Lambda = np.array(
            [
                [1 / (radius ** 2), 0, 0],
                [0, 1 / (radius ** 2), 0],
                [0, 0, 0]
            ]
        )

    def h(self, state):
        return (state - self.obstacle).T @ self.Lambda @ (state - self.obstacle) - 1
    
    def h_x(self, state):
        return 2 * (state - self.obstacle).T @ self.Lambda
    
    def h_xx(self, state):
        return 2 * self.Lambda
    
    def h_u(self, state, control):
        return np.zeros((1, 2))
