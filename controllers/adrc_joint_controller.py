import numpy as np
from observers.eso import ESO
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd
        self.model = ManiuplatorModel(Tp)

        A = np.array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        B = np.array([[0.], [self.b], [0.]])
        L = np.array([[3*p], [3*p**2], [p**3]])
        W = np.array([1., 0., 0.])
        self.eso = ESO(A, B, W, L, q0, Tp)
        self.prev_u = 0.

    def set_b(self, b):
        ### TODO update self.b and B in ESO
        self.b = b
        B = np.array([[0.], [self.b], [0.]])
        self.eso.set_B(B)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot, i):
        ### TODO implement ADRC
        control_input = np.array([self.prev_u])
        self.eso.update(x[i], control_input[:, np.newaxis])
        #control_input = np.array([self.prev_u])
        #self.eso.update(x[i], control_input[:, np.newaxis])
        z = self.eso.get_state()
        print("controller")
        print(z)
        q, q_dot, f = z
        u = (q_d_ddot + self.kd * (q_d_dot - q_dot) + self.kp * (q_d - q) - f) / self.b
        self.prev_u = u
        B = np.linalg.inv(self.model.M(x))
        self.set_b(B[i, i])
        return u
