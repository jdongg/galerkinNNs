import numpy as np
from Formulations.poisson2D.galerkinNN_poisson2D_H2weighted import GNN
import sys
np.set_printoptions(threshold=sys.maxsize)
from QuadratureRules.QuadratureRules import QuadratureRules
import os, pickle

RESULTS_PATH = 'poisson2D_Lshaped_extended'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

"""
This script solves the Poisson equation in 2D in the L-shaped domain.
To run, execute

        python -m Examples/poisson2D_Lshaped_lsq 

from the top-level directory, i.e. galerkinNNs.
"""
with open("data/Lshaped_reference_solution.pkl", 'rb') as file:
    loaded_data = pickle.load(file)

def source(x, y):
    f1 = np.ones(x.shape)
    return f1

def exact(x, y):
    u1e = np.zeros(x.shape)
    
    # load in fenicsx solutions. only evaluated at 
    # Gauss-Legendre (0,0) rules with 150 x 150 nodes
    # in each quadrant and uniformly spaced plotting points
    # with shape 251 x 251. 
    if (x.shape[0] == int(3*150*150)):
        u1e = loaded_data["solution_validation"]
    elif (x.shape[0] == int(251*251)):
        u1e = loaded_data["solution_stream"]

    return u1e

# exact gradient and second derivatives are not available
def exact_grad(x, y):
    u1xe = np.zeros(x.shape)
    u1ye = np.zeros(x.shape)
    return u1xe, u1ye

def main():

    MAX_EPOCH = 1000                # max epochs per basis function
    MAX_REFS = 6                    # number of basis functions
    INIT_NEURONS = 20               # width of hidden layers for first basis function
    INIT_SCALE = 1.0                # initial scaling parameter for activation (currently unused)
    SCALE_INCREMENT = 0.35          # linear incremenet for scaling, i.e. INIT_SCALE + SCALE_INCREMENT * k (currently unused)
    BASE_LEARNING_RATE = 1.e-2      # learning rate for first basis function
    ITER_DECAY_RATE = 1.15          # decay rate for lr, i.e. BASE_LEARNING_RATE / ITER_DECAY_RATE^k
    EXP_DECAY_RATE = 0.9            # decay rate for exponential decay rule
    IS_EXTENDED = True              # toggle extended NN architecture
    beta = 2.0 / 3.0 + 0.0          # weight for Sobolev spaces, i.e. r^beta

    tol = 1.e-9             # tolerance for number of basis functions; when estimator < tol, stop (currently unused)
    gradTol = 0.00000001    # tolerance for training; when relative change in loss < tol, stop
    boundarPen1 = 1.e3      # penalization for L2 norm on boundary
    boundarPen2 = 1.e0      # penalization for H1 seminorm on boundary

    # quadrature rules
    ng = 96
    QuadTrain = QuadratureRules()
    QuadTrain.GaussLegendreLshaped(ng)
    QuadVal = QuadratureRules()
    QuadVal.GaussLegendreLshaped(150)

    # length and width of bounding box for domain, e.g. L-shaped domain
    # is enclosed in the box (-1,1)^2
    Lx = 2.0 
    Ly = 2.0

    # uniformly spaced points for plotting routines
    nstream = 251
    x = np.linspace(-1.0, 1.0, nstream)
    y = np.linspace(-1.0, 1.0, nstream)
    X, Y = np.meshgrid(x, y)
    xStream = np.zeros([nstream**2, 2])
    xStream[:,0] = np.reshape(X, [nstream**2, ])
    xStream[:,1] = np.reshape(Y, [nstream**2, ])

    for i in range(nstream**2):
        if (xStream[i,0] > 0.0 and xStream[i,1] < 0.0):
            xStream[i,0] = np.nan
            xStream[i,1] = np.nan

    model = GNN(INIT_NEURONS, INIT_SCALE, Lx, Ly, IS_EXTENDED, QuadTrain, QuadVal, xStream, lambda x, y: source(x, y), 
                lambda x, y: exact(x, y), lambda x, y: exact_grad(x, y), tol, gradTol, boundarPen1, boundarPen2, SCALE_INCREMENT, 
                BASE_LEARNING_RATE, ITER_DECAY_RATE, EXP_DECAY_RATE, MAX_EPOCH, MAX_REFS, RESULTS_PATH, beta)
    model.generateBasis()

if __name__ == '__main__':
    main()