import numpy as np
from Formulations.stokes2D.galerkinNN_stokes2Dstationary_H2weighted import GNN
import sys
np.set_printoptions(threshold=sys.maxsize)
import os
from QuadratureRules.QuadratureRules import QuadratureRules

RESULTS_PATH = 'stokes2Dstationary_smooth'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

"""
This script solves the Stokes equation in 2D for a smooth manufactured solution 
(for testing convergence). To run, execute

        python -m Examples/stokes2Dstationary_manufactured_lsq

from the top-level directory, i.e. galerkinNNs.
"""
def source(x, y):
    f1 = np.zeros(x.shape)  # first component of momentum source, i.e. -L(u) + grad(p) = [f1, f2]^T
    f2 = np.zeros(x.shape)  # second component of momentum source
    f3 = np.zeros(x.shape)  # divergence source, i.e. div(u) = g
    f3x = np.zeros(x.shape) # derivatives of g for H1 penalization of divergence equation
    f3y = np.zeros(x.shape)

    for i in range(x.shape[0]):
        f1[i] = -(2.0) + y[i]
        f2[i] = -(-2.0 * np.sin(x[i] + y[i])) + x[i] 
        f3[i] = 1.0 + np.cos(x[i] + y[i])
        f3x[i] = -np.sin(x[i] + y[i]) 
        f3y[i] = -np.sin(x[i] + y[i]) 

    return f1, f2, f3, f3x, f3y


def exact(x, y):
    u1e = np.zeros(x.shape)
    u2e = np.zeros(x.shape)
    pe = np.zeros(x.shape)

    for i in range(x.shape[0]):
        u1e[i] = x[i] + y[i]**2 
        u2e[i] = np.sin(x[i] + y[i]) 
        pe[i] = x[i] * y[i] 

    return u1e, u2e, pe


# used only for H1 boundary condition
def exact_grad(x, y):

    u1xe = np.zeros(x.shape)
    u1ye = np.zeros(x.shape)
    u2xe = np.zeros(x.shape)
    u2ye = np.zeros(x.shape)
    pxe = np.zeros(x.shape)
    pye = np.zeros(x.shape)

    for i in range(x.shape[0]):
        u1xe[i] = 1.0
        u1ye[i] = 2.0 * y[i]
        u2xe[i] = np.cos(x[i] + y[i])
        u2ye[i] = np.cos(x[i] + y[i])
        pxe[i] = y[i] 
        pye[i] = x[i] 

    return u1xe, u1ye, u2xe, u2ye, pxe, pye


def main():

    ng = 72                 # number of quadrature nodes in shortest dimension of domain          
    tol = 1.e-9             # tolerance for stopping subspace augmentation, i.e. if <r(u_{i-1}), phi_i> < tol, stop (currently unused)
    gradTol = 0.000000001   # tolerance for stopping training, i.e. if relative change in loss < gradTol, stop training
    boundarPen1 = 2.e3      # penalty parameter for L2 norm on boundary and compatibility condition
    boundarPen2 = 2.e0      # penalty parameter for H1 seminorm on boundary
    boundarPen3 = 2.e3      # unused
    H1DivergenceFlag = 1    # flag for toggling H1 seminorm terms of divergence equation
    isExtended = False      # flag for toggling extended NN architecture
    withCutoff = False      # use cutoff function to isolate singular functions
    beta = 0.0              # weight for Sobolev spaces, i.e. r^beta

    MAX_EPOCH = 200             # maxmimum epochs for training each basis function
    MAX_REFS = 6                # maximum number of basis functions for subspace
    INIT_NEURONS = 20           # width of hidden layers for first basis function
    INIT_SCALE = 1.0            # currently unused
    SCALE_INCREMENT = 0.25      # currently unused
    BASE_LEARNING_RATE = 1.e-2  # initial learning rate for first basis function
    ITER_DECAY_RATE = 1.1       # BASE_LEARNING_RATE / ITER_DECAY_RATE^k is learning rate for kth basis function
    EXP_DECAY_RATE = 0.9        # decay rate of exponential schedule
        
    # length and width bounding box for domain (for Fourier feature mapping wave numbers)
    Lx = 4.0
    Ly = 5.0

    # quadrature and plotting points. 
    #       QuadTrain: rules for training step
    #       QuadVal: rules for computing errors
    #       xStream: plotting points on rectangular grid
    QuadTrain = QuadratureRules()
    QuadTrain.GaussLegendreTshaped(ng)
    QuadVal = QuadratureRules()
    QuadVal.GaussLegendreTshaped(120)

    nstream = 150
    x = np.linspace(-2.0, 2.0, nstream)
    y = np.linspace(-3.0, 2.0, nstream)
    X, Y = np.meshgrid(x, y)
    xStream = np.zeros([nstream**2, 2])
    xStream[:,0] = np.reshape(X, [nstream**2, ])
    xStream[:,1] = np.reshape(Y, [nstream**2, ])

    # set points outside cavity to NaN so they don't show in plots (very hack-y)
    for i in range(nstream):
        for j in range(nstream):
            idx = i * nstream + j
            if ((xStream[idx,1] < 3.0 * xStream[idx,0] - 3.0 and xStream[idx,1] < 0.0) or
                (xStream[idx,1] < -3.0 * xStream[idx,0] - 3.0 and xStream[idx,1] < 0.0)):
                xStream[idx,0] = np.nan
                xStream[idx,1] = np.nan

    model = GNN(INIT_NEURONS, INIT_SCALE, isExtended, withCutoff, Lx, Ly, QuadTrain, QuadVal, xStream, 
                lambda x, y: source(x, y), lambda x, y: exact(x, y), lambda x, y: exact_grad(x, y),
                tol, gradTol, boundarPen1, boundarPen2, boundarPen3, H1DivergenceFlag, 
                SCALE_INCREMENT, BASE_LEARNING_RATE, ITER_DECAY_RATE, EXP_DECAY_RATE, MAX_EPOCH, MAX_REFS, RESULTS_PATH, beta)
    model.generateBasis()

if __name__ == '__main__':
    main()