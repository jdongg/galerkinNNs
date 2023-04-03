import numpy as np
from formulations.galerkinNN_funcapprox2D import GNN
import scipy.interpolate
import sys
np.set_printoptions(threshold=sys.maxsize)

import os

RESULTS_PATH = 'funcapprox2D_results'

if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)


MAX_EPOCH = 150 	# max epochs per basis function
MAX_REFS = 8 		# number of basis functions
INIT_NEURONS = 20   # neurons for the first basis function
INIT_SCALE = 1.0    # adaptive scaling for the first basis function


# function to approximate
def exact(x, y):

	# this data comes from a suspension flow simulation. we interpolate to 
	# generate data at arbitrary arguments
	Data = np.loadtxt('data/volumefraction.dat')
	XY = Data[:,0:2]
	PHI = Data[:,2]

	U = scipy.interpolate.griddata(XY, PHI, (x,y), method='cubic')

	return U


def main():

	Ny = 256			  # training points on Cartesian grid in y-direction
	Nt = 128			  # " in t-direction
	tol = 1.e-9			  # try to get energy error less than tol
	gradTol = 0.0001      # stop training when relative change in loss < gradTol

	model = GNN(lambda x,y: exact(x,y), INIT_NEURONS, INIT_SCALE, Ny, Nt, tol, gradTol, 
				MAX_EPOCH, MAX_REFS, RESULTS_PATH)
	model.generateBasis()

if __name__ == '__main__':
	main()