import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp, m3=0.1, r3=0.05):
        #  na podstawie urdf
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3.
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2
        self.alpha = self.m1 * (self.l1/2) ** 2 + self.I_1 + self.m2*(self.l1**2+(self.l2/2)**2) + self.I_2 + self.m3*(self.l1**2+self.l2**2) + self.I_3
        self.beta = self.m2 * self.l1 * self.l2/2 + self.m3 * self.l1 * self.l2
        self.gamma = self.m2*(self.l2/2)**2 + self.I_2 + self.m3 * self.l2**2 + self.I_3

        self.Kp = np.array([[15, 0], [0, 15]])
        self.Kd = np.array([[10, 0], [0, 10]])

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        M11= self.alpha + 2*self.beta*np.cos(q2)
        M12= self.gamma + self.beta*np.cos(q2)
        M21= self.gamma + self.beta*np.cos(q2)
        M22= self.gamma

        return np.array([[M11, M12], [M21, M22]])

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        C11 = -self.beta*np.sin(q2)*q2_dot
        C12= -self.beta*np.sin(q2)*(q1_dot+q2_dot)
        C21= self.beta*np.sin(q2)*q1_dot
        C22=0
        return np.array([[C11, C12], [C21, C22]])
