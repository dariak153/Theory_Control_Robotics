import numpy as np
from .controller import Controller


class PDDecentralizedController(Controller):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def calculate_control(self, q, q_dot, q_d, q_d_dot, q_d_ddot):
        ### TODO: Please implement me
        error_pos=q_d-q
        error_vel=q_d_dot - q_dot
        u = self.kp * error_pos + self.kd * error_vel
        return u
