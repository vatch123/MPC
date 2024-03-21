from abc import abstractmethod, ABC
import matplotlib.pyplot as plt
import numpy as np

class Loss(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def p(self, state, control):
        """Loss value"""
        pass
    
    @abstractmethod
    def p_x_con(self, state):
        """Derivate of the continuous loss wrt state"""
        pass
    
    @abstractmethod
    def p_x_dis(self, state):
        """Derivative of the discrete loss wrt state"""
        pass

class QuadracticLoss(Loss):
    def __init__(self, Q, Q_bar, R_bar, final_pos=None):
        self.Q = Q
        self.Q_bar = Q_bar
        self.R_bar = R_bar
        self.loss_record = []

        if final_pos is None:
            self.final_pos = np.array([0, 0, 0])
        else:
            self.final_pos = final_pos

    def cont_loss(self, states):
        """Continuous Time loss"""
        try:
            # If states is a matrix of form N X k
            loss = np.diag((states - self.final_pos[None, :]).T @ self.Q @ (states - self.final_pos[None, :]))
        except ValueError:
            # If state is just a single state of dimension k
            loss = (states - self.final_pos).T @ self.Q @ (states - self.final_pos)
        return np.sum(loss)

    def discrete_loss(self, states):
        try:
            loss = np.diag((states - self.final_pos[None, :]).T @ self.Q_bar @ (states - self.final_pos[None, :]))
        except ValueError:
            loss = (states - self.final_pos).T @ self.Q_bar @ (states - self.final_pos)
        return np.sum(loss)

    def control_loss(self, control):
        try:
            loss = np.diag(control.T @ self.R_bar @ control)
        except ValueError:
            loss = control.T @ self.R_bar @ control
        return np.sum(loss)

    def p(self, states, control):
        """Value of the loss function."""
        self.loss_record.append([self.cont_loss(states), self.discrete_loss(states), self.control_loss(control)])
        return self.cont_loss(states) + self.discrete_loss(states) + self.control_loss(control)
    
    def p_x_dis(self, state):
        return 2 * self.Q_bar @ (state - self.final_pos)
    
    def p_x_con(self, state):
        return 2 * self.Q @ (state - self.final_pos)

    def plot_loss(self):
        self.loss_record = np.array(self.loss_record)
        plt.plot(self.loss_record[:, 0])
        plt.title('cont_loss')
        plt.show()
        plt.plot(self.loss_record[:, 1])
        plt.title('discrete_loss')
        plt.show()
        plt.plot(self.loss_record[:, 2])
        plt.title('control_loss')
        plt.show()
