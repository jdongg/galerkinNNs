import numpy as np
from Formulations.hemker1D.galerkinNN_hemker1D_H2weighted import GNN
import sys
np.set_printoptions(threshold=sys.maxsize)
sys.path.insert(0, '../QuadratureRules')
from QuadratureRules.QuadratureRules import QuadratureRules
import os, pickle
from scipy.special import erf

RESULTS_PATH = 'hemker1D_1e-5'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)


"""
This script solves the Hemker equation in 1D:

    ε * u''(x) + x * u'(x) = -ε * π^2 * np.cos(π * x) - π * x * np.sin(π * x)
    u(-1) = -2,   u(1) = 0

using a least squares variational formulation. To run, execute

        python -m Examples/hemker1D_lsq 

from the top-level directory, i.e. galerkinNNs.
"""
myeps = 1.e-5

def source(x):
    f1 = -myeps * np.pi**2 * np.cos(np.pi * x) - np.pi * x * np.sin(np.pi * x)
    return f1

def exact(x):
    u1e = np.cos(np.pi * x) + erf(x / np.sqrt(2*myeps)) / erf(1.0 / np.sqrt(2*myeps))
    return u1e

# exact gradient and second derivatives are not available
def exact_grad(x):
    u1xe = np.pi * np.cos(np.pi * x) + 5.0*np.pi * np.cos(5*np.pi*x) + 12.0*np.pi * np.cos(12*np.pi*x)
    return u1xe

def main():

    MAX_EPOCH = 2000                # max epochs per basis function (reduce as needed!)
    MAX_REFS = 7                    # number of basis functions
    INIT_NEURONS = 20               # width of hidden layers for first basis function
    INIT_SCALE = 1.0                # initial scaling parameter for activation (currently unused)
    SCALE_INCREMENT = 0.35          # linear incremenet for scaling, i.e. INIT_SCALE + SCALE_INCREMENT * k (currently unused)
    BASE_LEARNING_RATE = 1.e-2      # learning rate for first basis function
    ITER_DECAY_RATE = 1.15          # decay rate for lr, i.e. BASE_LEARNING_RATE / ITER_DECAY_RATE^k
    EXP_DECAY_RATE = 0.75            # decay rate for exponential decay rule
    IS_EXTENDED = False             # toggle extended NN architecture
    beta = 0.0 / 3.0 + 0.0          # weight for Sobolev spaces, i.e. r^beta

    tol = 1.e-9             # tolerance for number of basis functions; when estimator < tol, stop (currently unused)
    gradTol = 0.00000001    # tolerance for training; when relative change in loss < tol, stop
    boundarPen1 = 1.e3      # penalization for L2 norm on boundary
    boundarPen2 = 1.e0      # penalization for H1 seminorm on boundary

    # quadrature rules
    ng = 1024
    QuadTrain = QuadratureRules()
    QuadTrain.GaussLegendreInterval(ng, -1, 1)
    QuadVal = QuadratureRules()
    QuadVal.GaussLegendreInterval(1000, -1, 1)

    # length of domain (for Fourier feature mapping)
    Lx = 2.0 

    # uniformly spaced points for plotting routines
    nstream = 251
    x = np.linspace(-1.0, 1.0, nstream)
    xStream = np.zeros([nstream, 2])
    xStream[:,0] = np.reshape(x, [nstream,])

    model = GNN(INIT_NEURONS, INIT_SCALE, Lx, IS_EXTENDED, QuadTrain, QuadVal, xStream, lambda x: source(x), 
                lambda x: exact(x), lambda x: exact_grad(x), tol, gradTol, boundarPen1, boundarPen2, SCALE_INCREMENT, 
                BASE_LEARNING_RATE, ITER_DECAY_RATE, EXP_DECAY_RATE, MAX_EPOCH, MAX_REFS, RESULTS_PATH, beta, myeps)
    model.generateBasis()

if __name__ == '__main__':
    main()