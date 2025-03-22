import numpy as np
from Formulations.stokes2D.galerkinNN_stokes2Dstationary_H2weighted_singularitylearning import GNN
import sys
np.set_printoptions(threshold=sys.maxsize)
import os
from QuadratureRules.QuadratureRules import QuadratureRules

RESULTS_PATH = 'stokes2Dstationary_Tchannel'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

"""
This script solves the Stokes equation in 2D in a circular sector with
trainable singular functions (Example 4.2 from [1]). To run, execute

        python -m Examples/stokes2Dstationary_sector_singularitylearning_lsq

from the top-level directory, i.e. galerkinNNs.
"""
def source(x, y):
    f1 = np.zeros(x.shape)  # first component of momentum source, i.e. -L(u) + grad(p) = [f1, f2]^T
    f2 = np.zeros(x.shape)  # second component of momentum source
    f3 = np.zeros(x.shape)  # divergence source, i.e. div(u) = g
    f3x = np.zeros(x.shape) # derivatives of g for H1 penalization of divergence equation
    f3y = np.zeros(x.shape)

    return f1, f2, f3, f3x, f3y


# only used to specify boundary conditions.
def exact(x, y):
    pe = np.zeros(x.shape)

    omega = (np.pi + np.arccos(1.0/np.sqrt(10.0))) / 2.0
    t = np.mod(np.arctan2(y, x) + 2.0*np.pi, 2.0*np.pi)
    u1e = (-1.0/(2.0*omega) * t**2 + t) 
    u2e = ((-omega*2.0-1.0)/(omega*2.0)**2 * t**3 + t**2 + t)

    return u1e, u2e, pe


# used only for H1 boundary condition
def exact_grad(x, y):

    pxe = np.zeros(x.shape)
    pye = np.zeros(x.shape)

    omega = (np.pi + np.arccos(1.0/np.sqrt(10.0))) / 2.0
    t = np.mod(np.arctan2(y, x) + 2.0*np.pi, 2.0*np.pi)
    dtdx = -y/(x**2 + y**2)
    dtdy = x/(x**2 + y**2)

    u1xe = (-1.0 / (2.0 * omega) * 2.0 * t + 1.0) * dtdx 
    u1ye = (-1.0 / (2.0 * omega) * 2.0 * t + 1.0) * dtdy

    u2xe = ((-omega*2.0-1.0)/(omega*2.0)**2 * 3.0 * t**2 + 2.0 * t + 1.0) * dtdx
    u2ye = ((-omega*2.0-1.0)/(omega*2.0)**2 * 3.0 * t**2 + 2.0 * t + 1.0) * dtdy

    return u1xe, u1ye, u2xe, u2ye, pxe, pye


def main():

    ng = 72                 # number of quadrature nodes in shortest dimension of domain          
    tol = 1.e-9             # tolerance for stopping subspace augmentation, i.e. if <r(u_{i-1}), phi_i> < tol, stop (currently unused)
    gradTol = 0.000000001   # tolerance for stopping training, i.e. if relative change in loss < gradTol, stop training
    boundarPen1 = 2.e5      # penalty parameter for L2 norm on boundary and compatibility condition
    boundarPen2 = 2.e2      # penalty parameter for H1 seminorm on boundary
    boundarPen3 = 2.e3      # unused
    H1DivergenceFlag = 1    # flag for toggling H1 seminorm terms of divergence equation
    isExtended = True       # flag for toggling extended NN architecture
    beta = 4.0/3.0          # weight for Sobolev spaces, i.e. r^beta

    MAX_EPOCH = 200             # maxmimum epochs for training each basis function
    MAX_REFS = 1                # maximum number of basis functions for subspace
    INIT_NEURONS = 40           # width of hidden layers for first basis function
    INIT_SCALE = 1.0            # currently unused
    SCALE_INCREMENT = 0.25      # currently unused
    BASE_LEARNING_RATE = 1.e-2  # initial learning rate for first basis function
    ITER_DECAY_RATE = 1.1       # BASE_LEARNING_RATE / ITER_DECAY_RATE^k is learning rate for kth basis function
    EXP_DECAY_RATE = 0.9        # decay rate of exponential schedule
        
    # length and width bounding box for domain (for Fourier feature mapping wave numbers)
    Lx = 2.0
    Ly = 2.0

    # quadrature and plotting points. 
    #       QuadTrain: rules for training step
    #       QuadVal: rules for computing errors
    #       xStream: plotting points on rectangular grid
    theta = np.pi + np.arccos(1.0/np.sqrt(10.0))
    QuadTrain = QuadratureRules()
    QuadTrain.GaussLegendreCircularSector(ng, theta)
    QuadVal = QuadratureRules()
    QuadVal.GaussLegendreCircularSector(150, theta)

    nstream = 150
    r = np.linspace(0.0, 1.0, nstream)
    t = np.linspace(0.0, theta, nstream)
    R, T = np.meshgrid(r, t)
    X = R * np.cos(T)
    Y = R * np.sin(T)
    xStream = np.zeros([nstream**2, 2])
    xStream[:,0] = np.reshape(X, [nstream**2, ])
    xStream[:,1] = np.reshape(Y, [nstream**2, ])

    model = GNN(INIT_NEURONS, INIT_SCALE, isExtended, Lx, Ly, QuadTrain, QuadVal, xStream, 
                lambda x, y: source(x, y), lambda x, y: exact(x, y), lambda x, y: exact_grad(x, y),
                tol, gradTol, boundarPen1, boundarPen2, boundarPen3, H1DivergenceFlag, 
                SCALE_INCREMENT, BASE_LEARNING_RATE, ITER_DECAY_RATE, EXP_DECAY_RATE, MAX_EPOCH, MAX_REFS, RESULTS_PATH, beta)
    model.generateBasis()

if __name__ == '__main__':
    main()