from connie import Connie
import json
import os.path
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm


class Connie2de(Connie):
    domain = 10  # Domain of work [-domain..+domain]X[-domain..+domain]

    def __init__(self, filename='', cluster_count=-1):
        Connie.__init__(self)

        self.phi = lambda psi, x: ((psi[0] - x[:, 0]) ** 2 + (psi[1] - x[:, 1]) ** 2)
        self.PSI = lambda w, x: np.array([(x[:, 0] * w).sum(), (x[:, 1] * w).sum()]) / w.sum()

        if filename == '':
            return

        # Load data, if it has already been generated.
        if os.path.isfile(filename):
            self.load_data(filename)
            return

        # Create data
        if cluster_count != -1:
            Connie.CONNIE["GENERATION"]["CLUSTER"]["COUNT"] = cluster_count
        self.create_data()

        # Save data
        if not (filename == 'void'):
            self.save_data(filename)

    def visualize(self, what):

        # Preparations
        figure, axis = Connie.visualize(self, what)
        do_decorate_2d = True

        if what == 'input' or what == 'clusters':
            plt.scatter(self.x_n[:, 0], self.x_n[:, 1], self.w_n * 50, c='k', edgecolors='k')

        if what == 'classification':
            # Add outliers
            index = self.c_n[:, 0] == -1
            plt.scatter(self.x_n[index, 0], self.x_n[index, 1], self.w_n * 50, c='k', edgecolors='k')

            # Add the clusters
            for c in range(self.C):
                index = self.c_n[:, 0] == c
                color = cm.jet(int((c + 1.0) / (self.C + 1.0) * 255))
                plt.scatter(self.x_n[index, 0], self.x_n[index, 1], self.w_n * 50, c=color, edgecolors=color)

        if what == 'clusters':
            for c in range(self.C):
                color = cm.jet(int((c + 1.0) / (self.C + 1.0) * 255))
                plt.scatter(self.psi_c[c, 0], self.psi_c[c, 1], 100, c=color,
                            edgecolors=color, marker='x', linewidths=2)

        if what == 'fnc':
            colors = 1-self.f_nc.max(axis=1)
            plt.scatter(self.x_n[:, 0], self.x_n[:, 1], self.w_n * 50, c=colors, linewidths=0, cmap='bone')

        if what == 'pn':
            colors = 1-self.p_n
            plt.scatter(self.x_n[:, 0], self.x_n[:, 1], self.w_n * 50, c=colors, linewidths=0, cmap='bone')

        if do_decorate_2d:
            axis.add_patch(
                patches.Rectangle((-self.domain, -self.domain), 2 * self.domain, 2 * self.domain, fill=False))
            plt.axis([-1.05 * self.domain, 1.05 * self.domain, -1.05 * self.domain, 1.05 * self.domain])
            plt.xlabel('x0')
            plt.ylabel('x1')
            plt.grid()

        # Done
        plt.show(block=False)
        return self

    def seed(self):
        self.psi_c = np.random.rand(self.C, 2) * 2 * self.domain - self.domain

    def create_data(self):
        # Preparation
        self.C = Connie.CONNIE["GENERATION"]["CLUSTER"]["COUNT"]
        cluster_population = Connie.CONNIE["GENERATION"]["CLUSTER"]["POPULATION"]
        self.x_n = np.empty((0, 2))

        # Create cluster data
        for c in range(self.C):
            center = np.random.rand(2, 1) * 2 * self.domain - self.domain
            x = np.random.randn(cluster_population, 2)
            x[:, 0] += center[0]
            x[:, 1] += center[1]
            self.x_n = np.concatenate((self.x_n, x))

        # Create noise data
        x = np.random.randn(cluster_population, 2) * self.domain
        self.x_n = np.concatenate((self.x_n, x))

        # Filter data
        self.domain *= 1.2
        index = (self.x_n[:, 0] < self.domain) * (self.x_n[:, 0] > -self.domain) * (self.x_n[:, 1] < self.domain) * (
            self.x_n[:, 1] > -self.domain) == 1
        self.x_n = self.x_n[index, :]

        # Finish-off
        self.N = self.x_n.shape[0]
        self.w_n = np.random.rand(self.N, 1)

    def load_data(self, filename):
        # Load file
        contents = pickle.load(open(filename, 'rb'))

        # load contents
        self.C = contents['C']
        self.domain = contents['domain']
        self.x_n = contents['x_n']
        self.N = contents['N']
        self.w_n = contents['w_n']

    def save_data(self, filename):
        # Save contents
        contents = {'C': self.C, 'domain': self.domain, 'x_n': self.x_n, 'N': self.N, 'w_n': self.w_n}

        # Save to file
        pickle.dump(contents, open(filename, 'wb'))

    @staticmethod
    def initialize():
        Connie.initialize()

        # Set problem class-specific lambda
        with open('connie2de.json') as configuration_file:
            config_data = json.load(configuration_file)
            Connie.CONNIE['PENALTY'] = {}
            Connie.CONNIE['PENALTY']['LAMBDA'] = config_data['PENALTY']['PHI1']
            p_inf = config_data['PENALTY']['PINF']
            Connie.CONNIE['PENALTY']['U'] = p_inf / (1 - p_inf)
