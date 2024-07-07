import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        error_pos=x[:2] - q_r
        error_vel=x[2:] - q_r_dot
        v = q_r_ddot - self.model.Kp @ error_pos - self.model.Kd @ error_vel
        u = self.model.M(x) @ v + self.model.C(x) @ x[2:]
        return u
