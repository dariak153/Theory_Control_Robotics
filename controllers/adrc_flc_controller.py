import numpy as np

#from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
#from models.ideal_model import IdealModel
from models.manipulator_model import ManiuplatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        self.L = np.array([[3*p[0], 0], [0, 3*p[1]], [3*p[0]**2, 0], [0, 3*p[1]**2], [p[0]**3, 0], [0, p[1]**3]])
        W = np.array([[1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.]])
        self.A = np.array([[0., 0., 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]])
        self.B = np.zeros((6, 2))
        self.eso = ESO(self.A, self.B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        x = np.concatenate([q, q_dot])
        inv_M = np.linalg.inv(self.model.M(x.flatten()))
        inv_M_and_C = inv_M @ self.model.C(x.flatten())
        self.A[2:4, 2:4] = - inv_M_and_C
        self.B[2:4, :] = inv_M
        self.eso.A = self.A
        self.eso.B = self.B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        z = self.eso.get_state()
        z = z[:, np.newaxis]
        error_pos=x[:2] - q_d
        error_vel= x[2:] - q_d_dot
        v = q_d_ddot - self.Kp @ error_pos - self.Kd @ error_vel
        state = z[0:4].flatten()
        input_difference = v[:, np.newaxis] - z[4:]
        state_contribution = z[2:4]
        input_contribution = z[0:2]
        u = self.model.M(state) @ input_difference + self.model.C(state) @ state_contribution
        self.update_params(input_contribution, state_contribution)
        self.eso.update(x[:2], u)
        return u

