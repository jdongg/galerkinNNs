import numpy as np
from formulations.galerkinNN_poisson1D_H1 import GNN

import sys
np.set_printoptions(threshold=sys.maxsize)

import os

RESULTS_PATH = 'poisson1D_H1_results'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)


MAX_EPOCH = 50		# max epochs per basis function	
MAX_REFS = 8		# max number of basis functions
INIT_NEURONS = 5 	# width of network for first basis function
INIT_SCALE = 1.0 	# scaling parameter for first network: sigma(INIT_SCALE * t)
eps = 1.e-3 		# penalty for boundary conditions


# RHS of Poisson equation
def source(x):
	return (2.0*np.pi)**2 * np.sin(2.0*np.pi*x) + 0.1 * (25.0*np.pi)**2 * np.sin(25.0*np.pi*x)
	# return 2.0*np.ones(x.shape)


# true solution
def exact(x):
	A = -4.0*np.pi*eps / (2.0*eps+1.0)
	B = np.pi*eps * (10.0*eps+9.0) / (4.0*eps+2.0)

	return np.sin(2.0*np.pi*x) + 0.1 * np.sin(25.0*np.pi*x) + 5.0*np.pi*eps*(10.0*eps - 8.0*x + 9.0)/(20.0*eps+10.0) # + A*x + B
	# return x*(1.0-x) + eps


# derivative of true solution
def exactDx(x):
	A = -4.0*np.pi*eps / (2.0*eps+1.0)
	B = np.pi*eps * (10.0*eps+9.0) / (4.0*eps+2.0)

	return (2.0*np.pi) * np.cos(2.0*np.pi*x) + 0.1 * (25.0*np.pi) * np.cos(25.0*np.pi*x) - 40.0*np.pi*eps/(20.0*eps+10.0) #+ A
	# return 1.0 - 2.0*x 


def main():

	ng = 1024	 		  # number of training points
	xa = 0.0  	 		  # endpoints of domain
	xb = 1.0
	tol = 1.e-9  		  # try to get energy error less than tol
	gradTol = 0.001       # stop training when relative change in loss < gradTol
	boundaryPen = 1.0/eps # penalty parameter for BCs

	model = GNN(INIT_NEURONS, INIT_SCALE, lambda x: source(x), lambda x: exact(x),
			    lambda x: exactDx(x), xa, xb, ng, tol, gradTol, boundaryPen,
			    MAX_EPOCH, MAX_REFS, RESULTS_PATH)
	model.generateBasis()

if __name__ == '__main__':
	main()