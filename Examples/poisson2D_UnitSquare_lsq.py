import numpy as np
from Formulations.poisson2D.galerkinNN_poisson2D_H2weighted import GNN
import sys
np.set_printoptions(threshold=sys.maxsize)
from QuadratureRules.QuadratureRules import QuadratureRules
import os

RESULTS_PATH = 'poisson2D_UnitSquare_H2'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

"""
This script solves the Poisson equation in 2D for a smooth sinusoidal solution.
To run, execute

        python -m Examples/poisson2D_UnitSquare_lsq 

from the top-level directory, i.e. galerkinNNs.
"""
def source(x, y):
    f1 = np.zeros(x.shape)
    for i in range(x.shape[0]):
        f1[i] = np.sin(2.0 * np.pi * x[i]) * np.sin(2.0 * np.pi * y[i]) * (2.0 * np.pi)**2 * 2.0
    return f1

def exact(x, y):
    u1e = np.zeros(x.shape)

    for i in range(x.shape[0]):
        u1e[i] = np.sin(2.0 * np.pi * x[i]) * np.sin(2.0 * np.pi * y[i])
    return u1e

def exact_grad(x, y):
    u1xe = np.zeros(x.shape)
    u1ye = np.zeros(x.shape)
    for i in range(x.shape[0]):
        u1xe[i] = np.cos(2.0 * np.pi * x[i]) * np.sin(2.0 * np.pi * y[i]) * 2.0 * np.pi
        u1ye[i] = np.sin(2.0 * np.pi * x[i]) * np.cos(2.0 * np.pi * y[i]) * 2.0 * np.pi
    return u1xe, u1ye

def main():
    MAX_EPOCH = 200             # max epochs per basis function
    MAX_REFS = 7                # number of basis functions
    INIT_NEURONS = 20           # width of first network
    INIT_SCALE = 1.0            # scaling for first activation function (no longer used)
    SCALE_INCREMENT = 0.35      # growth rate for scaling; INIT_SCALE + k * SCALE_INCREMENT (no longer used)
    BASE_LEARNING_RATE = 1.e-2  # learning rate for first basis
    ITER_DECAY_RATE = 1.1       # decay rate for ith basis; BASE_LEARNING_RATE / ITER_DECAY_RATE^k
    EXP_DECAY_RATE = 0.9        # exponential decay rate for learning rate
    IS_EXTENDED = False         # toggles extended NN architecture
    beta = 0.0                  # weight for Sobolev norms; r^beta

    ng = 95               # number of quadrature nodes in 1D (ng x ng total)
    tol = 1.e-9           # stop generating basis functions when estimate is < tol (currently unused)
    gradTol = 0.00000001  # stop training when relative change in loss is < gradTol
    boundarPen1 = 1.e3    # penalty parameter for L^2 norm on boundary
    boundarPen2 = 1.e1    # penalty parameter for H^1 seminorm on boundary

    # domain
    QuadTrain = QuadratureRules()
    QuadTrain.GaussLegendreUnitSquare(ng)
    QuadVal = QuadratureRules()
    QuadVal.GaussLegendreUnitSquare(150)

    nstream = 150
    x = np.linspace(0.0, 1.0, nstream)
    y = np.linspace(0.0, 1.0, nstream)
    X, Y = np.meshgrid(x, y)
    xStream = np.zeros([nstream**2, 2])
    xStream[:,0] = np.reshape(X, [nstream**2, ])
    xStream[:,1] = np.reshape(Y, [nstream**2, ])

    # length of bounding rectangle of domain (for initializing wave numbers 
    # for Fourier feature map)
    Lx = 1.0 
    Ly = 1.0

    model = GNN(INIT_NEURONS, INIT_SCALE, Lx, Ly, IS_EXTENDED, QuadTrain, QuadVal, xStream, lambda x, y: source(x, y), 
                lambda x, y: exact(x, y), lambda x, y: exact_grad(x, y), tol, gradTol, boundarPen1, boundarPen2, SCALE_INCREMENT, 
                BASE_LEARNING_RATE, ITER_DECAY_RATE, EXP_DECAY_RATE, MAX_EPOCH, MAX_REFS, RESULTS_PATH, beta)
    model.generateBasis()

if __name__ == '__main__':
    main()