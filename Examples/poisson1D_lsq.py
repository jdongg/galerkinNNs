import numpy as np
from Formulations.poisson1D.galerkinNN_poisson1D_H2weighted import GNN
import sys
np.set_printoptions(threshold=sys.maxsize)
sys.path.insert(0, '../QuadratureRules')
from QuadratureRules.QuadratureRules import QuadratureRules
import os

"""
This script solves the Poisson equation in 1D for a smooth sinusoidal solution.
To run, execute

        python -m Examples/poisson1D_lsq 

from the top-level directory, i.e. galerkinNNs.
"""
RESULTS_PATH = 'poisson1D_smooth'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

def source(x):
    f1 = (np.pi**2) * np.sin(np.pi * x) + (5.0*np.pi)**2 * np.sin(5*np.pi*x) + (12.0*np.pi)**2 * np.sin(12*np.pi*x)
    return f1

def exact(x):
    u1e = np.sin(np.pi * x) + np.sin(5*np.pi*x) + np.sin(12*np.pi*x)
    return u1e

# exact gradient is not available and not used anywhere by GNN
def exact_grad(x):
    u1xe = np.pi * np.cos(np.pi * x) + 5.0*np.pi * np.cos(5*np.pi*x) + 12.0*np.pi * np.cos(12*np.pi*x)
    return u1xe

def main():

    MAX_EPOCH = 1000                # max epochs per basis function (reduce as needed!)
    MAX_REFS = 6                    # number of basis functions
    INIT_NEURONS = 20               # width of hidden layers for first basis function
    INIT_SCALE = 1.0                # initial scaling parameter for activation (currently unused)
    SCALE_INCREMENT = 0.35          # linear incremenet for scaling, i.e. INIT_SCALE + SCALE_INCREMENT * k (currently unused)
    BASE_LEARNING_RATE = 1.e-2      # learning rate for first basis function
    ITER_DECAY_RATE = 1.15          # decay rate for lr, i.e. BASE_LEARNING_RATE / ITER_DECAY_RATE^k
    EXP_DECAY_RATE = 0.9            # decay rate for exponential decay rule
    IS_EXTENDED = False             # toggle extended NN architecture
    beta = 0.0 / 3.0 + 0.0          # weight for Sobolev spaces, i.e. r^beta

    tol = 1.e-9             # tolerance for number of basis functions; when estimator < tol, stop (currently unused)
    gradTol = 0.00000001    # tolerance for training; when relative change in loss < tol, stop
    boundarPen1 = 1.e3      # penalization for L2 norm on boundary
    boundarPen2 = 1.e0      # penalization for H1 seminorm on boundary

    # quadrature rules
    ng = 128
    QuadTrain = QuadratureRules()
    QuadTrain.GaussLegendreInterval(ng, -1, 1)
    QuadVal = QuadratureRules()
    QuadVal.GaussLegendreInterval(150, -1, 1)

    # length of domain (for Fourier feature mapping)
    Lx = 2.0 

    # uniformly spaced points for plotting routines
    nstream = 251
    x = np.linspace(-1.0, 1.0, nstream)
    xStream = np.zeros([nstream, 2])
    xStream[:,0] = np.reshape(x, [nstream,])

    model = GNN(INIT_NEURONS, INIT_SCALE, Lx, IS_EXTENDED, QuadTrain, QuadVal, xStream, lambda x: source(x), 
                lambda x: exact(x), lambda x: exact_grad(x), tol, gradTol, boundarPen1, boundarPen2, SCALE_INCREMENT, 
                BASE_LEARNING_RATE, ITER_DECAY_RATE, EXP_DECAY_RATE, MAX_EPOCH, MAX_REFS, RESULTS_PATH, beta)
    model.generateBasis()

if __name__ == '__main__':
    main()