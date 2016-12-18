import numpy as np
import numpy.matlib
import json
import matplotlib.pyplot as plt
import math
import datetime


class Connie:
    CONNIE = {}

    def __init__(self):
        # Controls
        self.silent = True

        self.C = 16  # Number of clusters
        self.N = 100  # Number of entities
        self.x_n = np.zeros(1)  # Data x_n[n,:] [N X ?]
        self.w_n = np.zeros(1)  # Weights w_n[n] [N X 1]

        # Functions
        self.u = lambda x: x / (1 + x)
        self.u_prime = lambda x: 1 / ((1 + x) ** 2)
        self.phi = lambda psi, x: np.zeros(1)  # Distance function between a class and an entity
        self.PSI = lambda w, x: np.zeros(1)  # Cluster fitting function

        # Outputs
        self.psi_c = np.zeros(1)  # Cluster information psi_c[c,:] [C X ?]
        self.c_n = np.zeros(1)  # Classification results _n(c) [C X 1]

        # Internals
        self.f_nc = np.zeros(1)  # Fuzzy membership (internal) [N X C]
        self.p_n = np.zeros(1)  # Possibilistic membership values [N x_n 1]

    def visualize(self, what):
        figure = plt.figure(what)
        figure.clear()
        axis = figure.add_subplot(111, aspect='equal')

        return figure, axis

    def execute(self):
        # Preparation
        t_start = datetime.datetime.now()
        if self.w_n.shape[0] is 0:
            self.w_n = np.ones((self.N, 1))
        self.f_nc = np.zeros((self.N, self.C))
        self.p_n = np.zeros((self.N, 1))
        phi_nc = np.zeros((self.N, self.C))
        outlier_cost = Connie.CONNIE['PENALTY']['U']
        scale = Connie.CONNIE['PENALTY']['LAMBDA']

        # Prepare seeds
        self.seed()

        # Loop
        delta_previous = 0
        iteration = 0
        if not self.silent:
            print "Iteration:"
        while True:
            # Calculate phi_nc and u_nc
            for c in range(self.C):
                phi_nc[:, c] = self.phi(self.psi_c[c, :], self.x_n)
            u_nc = self.u(phi_nc / scale)

            # Calculate self.f_nc
            unc_1 = 1.0 / u_nc
            # noinspection PyUnresolvedReferences
            den = np.matlib.repmat(unc_1.sum(axis=1), self.C, 1).transpose()
            self.f_nc = unc_1 / den

            # Calculate p_n
            self.p_n[:, 0] = 1.0 / (((self.f_nc ** 2) * u_nc).sum(axis=1) / outlier_cost + 1)

            # Calculate w_nc
            w_nc = np.matlib.repmat(self.w_n, 1, self.C) * \
                self.f_nc ** 2 * \
                np.matlib.repmat(self.p_n, 1, self.C) ** 2 * \
                self.u_prime(phi_nc / scale)

            # Update psi_c
            for c in range(self.C):
                self.psi_c[c, :] = self.PSI(w_nc[:, c], self.x_n)

            # Calculate delta
            delta = (self.w_n *
                     (
                         self.p_n ** 2 * (self.f_nc ** 2 * u_nc).sum() +
                         (1 - self.p_n) ** 2 * outlier_cost / self.C
                     ).sum()).sum()

            # Decide if convergence is achieved
            iteration += 1
            if iteration > Connie.CONNIE['CONVERGENCE']['ITERATION']:
                if not self.silent:
                    print "Too many iterations. Quitting."
                break
            if iteration >= Connie.CONNIE['CONVERGENCE']['GRACE']:
                delta_change = math.fabs(delta - delta_previous) / delta
                delta_previous = delta
                if not self.silent:
                    print " : {0} - Change: {1}".format(iteration, delta_change)
                if delta_change < Connie.CONNIE['CONVERGENCE']['DELTA']:
                    if not self.silent:
                        print "Converged."
                    break

        # Classify
        self.classify()
        t_end = datetime.datetime.now()

        # Done
        if not self.silent:
            print "{0} milliseconds".format(int((t_end - t_start).total_seconds() * 1000))
        return self

    def classify(self):
        # Calculate f_max
        f_max = self.f_nc.max(axis=1)
        f_max[self.p_n[:, 0] < 0.5] = 0

        # Calculate self.c_n
        self.c_n = -np.ones((self.N, 1))
        for c in range(self.C):
            self.c_n[self.f_nc[:, c] == f_max] = c

    @staticmethod
    def initialize():
        with open('connie.json') as configuration_file:
            Connie.CONNIE = json.load(configuration_file)

    def seed(self):
        pass
