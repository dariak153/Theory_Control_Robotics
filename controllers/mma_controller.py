import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.Tp = Tp
        self.models = []
        self.m3_values = [0.1, 0.01, 1.0]
        self.r3_values = [0.05, 0.01, 0.3]
        for i in range(len(self.m3_values)):
            self.models.append(ManiuplatorModel(Tp, self.m3_values[i], self.r3_values[i]))

        self.i = 0
        self.prev_x = np.zeros(4)
        self.u = np.zeros(2)

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        errors = []
        for model in self.models:
            M = model.M(self.prev_x)
            C = model.C(self.prev_x)
            invM = np.linalg.inv(M)
            zeros = np.zeros((2, 2), dtype=np.float32)
            A_top = np.concatenate((zeros, np.eye(2)), axis=1)
            A_bottom = np.concatenate((zeros, -invM @ C), axis=1)
            A = np.concatenate((A_top, A_bottom), axis=0)
            b_top = zeros
            b_bottom = invM
            b = np.concatenate((b_top, b_bottom), axis=0)
            x_dot = A @ self.prev_x[:, np.newaxis] + b @ self.u
            calculated_x = self.prev_x + x_dot.T * self.Tp
            errors.append(np.linalg.norm(calculated_x[0] - x))

        self.i = np.argmin(errors)
        print(errors)
        print(self.i)



    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        error_pos=x[:2]-q_r
        error_vel=x[2:]-q_r_dot
        v = q_r_ddot
        v = q_r_ddot - self.models[self.i].Kp @ error_pos - self.models[self.i].Kd @ error_vel
        self.u = np.array(self.models[self.i].M(x) @ v[:, np.newaxis] + self.models[self.i].C(x) @ q_dot[:, np.newaxis])
        self.prev_x = x
        return self.u
