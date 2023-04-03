import jax.numpy as jnp
from jax.example_libraries import optimizers
import jax
from joblib import Parallel, delayed
import numpy as np
import math
import pickle

import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import scipy.io as sio
from scipy.special import roots_jacobi
import scipy
from functools import reduce

from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)


#------------------------------------------------------------------
# class GNN
#
# Purpose: this class creates a Galerkin neural network and uses
#		   it to approximate a 2D function f:
#				a(u,v) = (u,v)
#				L(v) = (f,v)
#
# Parameters:
#		neurons -- width of the neural network for the first basis
#				   function. widths for subsequent networks follow
#				   the rule neurons * 20^{i-1}
#		scale -- initial scaling argument for the activation function
#						sigma(scale * t).
#				 subsequent activation functions follow the rule
#				 scale + 2^{i-1}
#		u -- function for true solution 
#		ng -- number of nodes for training
#		tolerance -- for generating basis functions. stoping computing
#					 new basis functions when 
#						<r(u_{i-1}), \varphi_i^NN> < tolerance
#		gradTolerance -- for minimizing loss function. stop when 
#							|loss_i - loss_{i-1}| / |loss_{i-1}| < gradTolerance
#		maxEpoch -- maximum number of iterations to train each 
#					basis function for
#		maxRef -- maximum number of basis functions to compute
#		RESULTS_PATH -- directory for saving results
#
#------------------------------------------------------------------
class GNN:
	def __init__(self, u, neurons, scale, Ny, Nt, tolerance, gradTolerance, 
				 maxEpoch, maxRef, RESULTS_PATH):

		# path for saving results
		self.RESULTS_PATH = RESULTS_PATH

		# function for evaluating exact function
		self.exact = u

		# generate quadrature rule in interior; training
		self.Ny, self.Nt = Ny, Nt
		self.xGlobal, self.wGlobal, self.wCompat = self.gaussLegendre(self.Ny, self.Nt)

		y = np.reshape(self.xGlobal[:,0], [self.Ny, self.Nt])
		t = np.reshape(self.xGlobal[:,1], [self.Ny, self.Nt])

		self.phiTrain = np.reshape(self.exact(y, t), [self.Ny*self.Nt, 1])

		# generate quadrature rule in interior; validation
		self.NyVal, self.NtVal = 300, 150
		self.xGlobalVal, self.wGlobalVal, _ = self.gaussLegendre(self.NyVal, self.NtVal)

		y = np.reshape(self.xGlobalVal[:,0], [self.NyVal, self.NtVal])
		t = np.reshape(self.xGlobalVal[:,1], [self.NyVal, self.NtVal])

		self.phiVal = np.reshape(self.exact(y, t), [self.NyVal*self.NtVal, 1])
		
	
		# list of basis functions of their derivatives
		self.basis = []
		self.basis_val = []

		# tolerance for adaptive basis generation
		self.tol = tolerance
		self.gradTol = gradTolerance

		# initial network width and argument scaling
		self.neurons = neurons
		self.scale = scale

		# max epochs and number of basis functions
		self.maxEpoch = maxEpoch
		self.maxRef = maxRef

		# results epochs
		self.L2_epoch = []
		self.Loss_epoch = []


		self.DOFS_iter = []
		self.L2_iter = []
		self.Loss_iter = []

		# holders for trained parameters for each basis function
		self.trainedParams = []


	#------------------------------------------------------------------
	# Function gaussLegendre
	#
	# Purpose: generate a tensor product Gaussian quadrature rule over
	#			the space-time domain (5, 35) x (1, 200). a Gauss-Legendre
	#			rule is used (does not include endpoints).
	#
	# Parameters:
	#		Ny -- number of nodes in the y-direction.
	#		Nt -- number of nodes in the strain-direction.
	#
	# Returns:
	#		xGlobal -- (Ny*Nt) x 2 array of quadrature nodes. the first 
	#					column contains y-coordinates and the second
	#					column contains strain coordinates.
	#		wGlobal -- (Ny*Nt) x 1 array of quadrature weights.
	#		wt -- quadrature weights for the one-dimensional quadrature
	#				over (1,200). this rule is necessary for implementation
	#				of the boundary condition
	#------------------------------------------------------------------
	def gaussLegendre(self, Ny, Nt):

		y, wy = roots_jacobi(Ny, 0.0, 0.0) 
		y = np.reshape(0.5*(35.0-5.0)*y + 0.5*(35.0+5.0), [Ny,1])
		wy = np.reshape(0.5*(35.0-5.0)*wy, [Ny,1])

		t, wt = roots_jacobi(Nt, 0.0, 0.0) 
		t = np.reshape(0.5*(200.0-1.0)*t + 0.5*(200+1.0), [Nt,1])
		wt = np.reshape(0.5*(200.0-1.0)*wt, [Nt,1])


		xGlobal = np.zeros([Ny*Nt, 2])
		wGlobal = np.zeros([Ny*Nt, 1])
		for i in range(Ny):
			for j in range(Nt):
				idx = i*Nt + j
				xGlobal[idx,0] = y[i]
				xGlobal[idx,1] = t[j]
				wGlobal[idx,0] = wy[i] * wt[j]

		return xGlobal, wGlobal, wt


	#------------------------------------------------------------------
	# Function initLayer
	#
	# Purpose: initialize weights and biases according to the box 
	# 			box initialization of Cyr, et. al.
	#
	# Parameters:
	#		m -- input dimension of weights
	#		n -- output dimension of weights
	#		L -- number of hidden layers in the network
	#		ell -- current layer number (for ResNet only)
	#
	# Returns:
	#		W -- weights of the hidden layer
	#		b -- biases of the hidden layer
	#		Wr -- residual weights of the hidden layer (only used for ResNets -
	#				currently not in use)
	#------------------------------------------------------------------
	def initLayer(self, m, n, L, ell):

		if (m == 2):
			# this is the box initialization
			p = np.array([[200.0, 40.0]]) * np.array(np.random.uniform(size=[n, m]))
			norm = np.array(np.random.normal(size=[n, m]))

			nhat = np.zeros([n, m])
			pmax = np.zeros([n, m])
			k = np.zeros([1, n])

			W = np.zeros([n, m])
			b = np.zeros([n, 1])

			for i in range(n):
				for j in range(m):
					nhat[i,j] = norm[i,j] / np.linalg.norm(norm[i,:], 2)
					pmax[i,j] = np.maximum(-1.0, np.sign(nhat[i,j]))

			for i in range(n):
				k[0,i] = 1.0 / np.abs(1.0 * np.sum((pmax[i,:] - p[i,:]) * nhat[i,:]))
				for j in range(m):
					W[i,j] = k[0,i]*nhat[i,j]

				b[i] = -k[0,i]*np.sum(nhat[i,:] * p[i,:])


			# this is the initialization described in Ainsworth and Dong (2021)
			W = np.zeros([n, m])
			b = np.zeros([n, 1])

			stop = int(math.floor(n/4))

			b[0:stop] = -np.reshape(np.linspace(1.0, 200.0, stop), [stop,1])
			b[stop:2*stop] = -np.reshape(np.linspace(0.0, 200.0, stop), [stop,1])
			b[2*stop:3*stop] = -np.reshape(np.linspace(5.0, 35.0, stop), [stop,1])
			b[3*stop:n] = np.reshape(np.linspace(0.0, 200.0, n-3*stop), [n-3*stop,1])

			tol = 0.00001
			W[0:stop,0] = np.ones(stop)
			W[stop:2*stop,0] = np.ones(stop)
			W[2*stop:3*stop,0] = np.zeros(stop)
			W[3*stop:n,0] = -np.ones(n-3*stop)

			W[0:stop,1] = np.zeros(stop)
			W[stop:2*stop,1] = np.ones(stop)
			W[2*stop:3*stop,1] = np.ones(stop)
			W[3*stop:n,1] = np.ones(n-3*stop)

			W = W.T
			b = b.T
		

		# other hidden layers
		else:
			# W = np.random.uniform(-1.0/np.sqrt(m), 1.0/np.sqrt(m), [m, n])
			# b = np.random.uniform(-1.0/np.sqrt(m), 1.0/np.sqrt(m), [1, n])
			# Wr = np.random.uniform(-1.0/np.sqrt(m), 1.0/np.sqrt(m), [m, n])

			mscale = (1.0 + 1.0/(L-1.0))**ell

			p = m * np.array(np.random.uniform(size=[n, m]))
			norm = np.array(np.random.normal(size=[n, m]))

			nhat = np.zeros([n, m])
			pmax = np.zeros([n, m])
			k = np.zeros([1, n])

			W = np.zeros([n, m])
			b = np.zeros([n, 1])

			for i in range(n):
				for j in range(m):
					nhat[i,j] = norm[i,j] / np.linalg.norm(norm[i,:], 2)
					pmax[i,j] = mscale * np.maximum(0.0, np.sign(nhat[i,j]))

			for i in range(n):
				k[0,i] = 1.0 / np.abs(np.sum((pmax[i,:] - p[i,:]) * nhat[i,:] * (L-1)))
				# print(k[i])
				for j in range(m):
					W[i,j] = k[0,i]*nhat[i,j]

				b[i] = -k[0,i]*np.sum(nhat[i,:] * p[i,:])

			W = W.T
			b = b.T
			
		return W, b


	#------------------------------------------------------------------
	# Function initParams
	#
	# Purpose: creates list of parameters for each layer of the network
	#
	# Parameters:
	#		sizes -- size[i] and size[i+1]; the input and output shape of
	#				layer i, 
	#		a -- left endpoint of computational domain
	#		b -- right endpoint of computational domain
	#
	# Returns:
	#		params -- list of parameters for each hidden layer;
	#					W, b = params[i].
	#------------------------------------------------------------------
	def initParams(self, sizes):
		return [self.initLayer(m, n, len(sizes), i) for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:]))]


	#------------------------------------------------------------------
	# Function createNetwork
	#
	# Purpose: return the output of the neural network.
	#
	# Parameters:
	#		T -- t value; must be in shape N x 1
	#		Y -- y value; must be in shape N x 1
	#		params -- list of the hidden parameters of each layer
	#		C -- activation weights/coefficients. these are obtained by
	#			calling self.galerkinUpdatePhi or self.galerkinUpdateSigma.
	#		scale -- scaling parameter for the activation function. larger
	#				values induce sharper gradients of the activation.
	#
	# Returns:
	#		U -- output of the neural network in shape N x 1.
	#------------------------------------------------------------------	
	@partial(jax.jit, static_argnums=(0,))
	def createNetwork(self, T, Y, params, C, scale):

		# first hidden layer
		W, b = params[0]
		Wt = jnp.reshape(W[0,:], [1, W.shape[1]])
		Wy = jnp.reshape(W[1,:], [1, W.shape[1]])

		H = jax.nn.tanh(scale * (T*Wt + Y*Wy + b)) 
		Hend = H

		for W, b in params[1:]:
			H = jax.nn.tanh(scale * (jnp.matmul(Hend, W) + b))
			Hend = H

		U = Hend @ C

		return U


	#------------------------------------------------------------------
	# Function networkArray
	#
	# Purpose: return an array of each individual activation function; e.g.
	#			for a network with one hidden layer, 
	#
	# Parameters:
	#		T -- t value; must be in shape N x 1
	#		Y -- y value; must be in shape N x 1
	#		params -- list of the hidden parameters of each layer
	#		C -- activation weights/coefficients. these are obtained by
	#			calling self.galerkinUpdatePhi or self.galerkinUpdateSigma.
	#		scale -- scaling parameter for the activation function. larger
	#				values induce sharper gradients of the activation.
	#
	# Returns:
	#		U -- output of the neural network in shape N x 1.
	#------------------------------------------------------------------	
	@partial(jax.jit, static_argnums=(0,))
	def networkArray(self, T, Y, params, scale):

		W, b = params[0]
		Wt = jnp.reshape(W[0,:], [1, W.shape[1]])
		Wy = jnp.reshape(W[1,:], [1, W.shape[1]])

		H = jax.nn.tanh(scale * (T*Wt + Y*Wy + b)) 
		Hend = H

		for W, b in params[1:]:
			H = jax.nn.tanh(scale * (jnp.matmul(Hend, W) + b))
			Hend = H

		return Hend



	#------------------------------------------------------------------
	# Function activMatrix
	#
	# Purpose: updates the stiffness matrix K and RHS F for computing
	#			the weights of the activation layer. The stiffness matrix 
	#			K has entries given by
	#			
	#				K_ij = a(sigma_i, sigma_j) 
	#
	#			where sigma_i := tanh(xGlobal * W_i + b_i) for a network
	#			with one hidden layer (the function is agnostic to the
	#			number of hidden layers and does not need to be modified 
	#			if deeper networks are used). 
	#
	#			The load vector F has entries given by
	#	
	#				F_i = L(sigma_i) - a(u_{i-1}, sigma_i),
	#	
	#			where phi_{i-1} is the (i-1) approximation to the function.
	#
	# Parameters:
	#		i -- current row of K/element of F to modify. this function
	#			is called in parallel since each row may be modified 
	#			independently of the others.
	#		NNoutput -- an N x n array in which the ith column contains
	#					the ith activation function evaluated at the quadrature
	#					points in xGlobal...for a network with one hidden layer,
	#					this is given by
	#						
	#						tanh(xGlobal * W_i + b_i).
	#
	#		u0 -- the (i-1) approximation to the PDE. For i=1, this is
	#				the initial guess the GNN method, which is always
	#				taken as 0 for all examples in this project.
	#		K -- the n x n stiffness matrix, where n is the number of 
	#			neurons in the last hidden layer.
	#		F -- the n x 1 RHS load vector.
	#
	#
	# Returns:
	#		in-line modification of K and F
	#------------------------------------------------------------------	
	def activMatrix(self, i, NNoutput, u0, K, F):

		# the ith activation function
		NN_idx = np.reshape(NNoutput[:,i], [self.xGlobal.shape[0], 1])

		# all of the activation functions (for vectorized computation)
		NN_jdx = NNoutput

		# vectorized computation of \int sigma_i * sigma_j using 
		# quadrature rule
		K[i,:] = np.sum(self.wGlobal * (NN_idx) * (NN_jdx), axis=0)
		
	 
	 	# computation of \int \phi^{FCM} * sigma_i
		F[i] = np.sum(self.wGlobal * (NN_idx) * (self.phiTrain))

		# computation of a(\phi_{i-1}, sigma_i) = \int u_{i-1} * sigma_i
		F[i] -= np.sum(self.wGlobal * (NN_idx) * (u0))

		return


	#------------------------------------------------------------------
	# Function galerkinUpdate
	#
	# Purpose: returns the weights in the activation layer of the neural
	#			network.
	#
	# Parameters:
	#		neurons -- neurons in the last hidden layer.
	#		params -- list of parameters of each layer in the network.
	#		u0 -- the (i-1) approximation to the problem.
	#		scale -- scaling parameter for the activation function. larger
	#				values induce sharper gradients of the activation.
	#
	# Returns:
	#		c -- the weights of the activation layer in shape n x 1.
	#------------------------------------------------------------------	
	def galerkinUpdate(self, neurons, params, u0, scale):

		N = self.xGlobal.shape[0]
		K = np.zeros([neurons, neurons])
		F = np.zeros([neurons, 1])

		# evaluate activation functions
		tTrain = np.reshape(self.xGlobal[:,1], [N,1])
		yTrain = np.reshape(self.xGlobal[:,0], [N,1])
		NNoutput = self.networkArray(tTrain, yTrain, params, scale)

		# assemble matrices, columns can be done in parallel
		Parallel(n_jobs=8, backend="threading")(
			delayed(self.activMatrix)(i, NNoutput, u0, K, F) for i in range(neurons))

		c, _, _, _ = scipy.linalg.lstsq(K, F, cond=None)
		# c = np.linalg.solve(K, F)
		c = np.reshape(c, [neurons,1])

		return c


	#------------------------------------------------------------------
	# Function computeLoss
	#
	# Purpose: returns the value of the loss function for the volume
	#			fraction:
	#
	#				Loss := L(v) - a(u_{i-1}, v)
	#						-----------------------.
	#						     a(v,v)^{1/2}
	#
	# Parameters:
	#		params -- list of parameters of each layer in the network.
	#		T -- N x 1 array of strain data to evaluate neural networks at
	#			(these points should correspond to a quadrature rule).
	#		Y -- N x 1 array of spatial data along the channel (these points
	#			should correspond to a quadrature rule).
	#		W -- N x 1 array of the quadrature rule weights for evaluating
	#			integrals in the loss function.
	#		coeffs -- weights of the activation layer of the network obtained
	#				from self.galerkinUpdatePhi.
	#		scalek -- scaling parameter for the activation function. larger
	#				values induce sharper gradients of the activation.
	#		U0 -- N x 1 array of the (i-1) approximation to the function
	#			at the N x 1 points (T,Y).
	#		F -- N x 1 array of the exact function evaluated at (T,Y).
	#
	# Returns:
	#		Loss -- value of the loss function.
	#------------------------------------------------------------------	
	@partial(jax.jit, static_argnums=(0,))
	def computeLoss(self, params, T, Y, W, coeffs, scalek, U0, F):
		
		# compute the output neural network for the current basis function
		phi = self.createNetwork(T, Y, params, coeffs, scalek)

		# compute a(v,v)^{1/2} in the loss function
		normint = jnp.sum(jnp.multiply(W, jnp.square(phi)))
		norm = jnp.sqrt(normint)
		
		# compute L(v) in the loss function
		r1 = jnp.sum(jnp.multiply(W, jnp.multiply(U0, phi)))

		# compute a(u_{i-1}, v) in the loss function
		r2 = jnp.sum(jnp.multiply(W, jnp.multiply(F, phi)))

		res = r2 - r1

		# max L = min -L since we are solving a maximization problem
		# but calling jax's minimization routines
		Loss = jnp.multiply(-1.0, jnp.abs(jnp.divide(res, norm)))

		return Loss



	#------------------------------------------------------------------
	# Function computeError
	#
	# Purpose: compute the L2 error in the current approximation to the
	#			volume fraction.
	#
	# Parameters:
	#		u0 -- N x 1 array of the ith approximation to function
	#			at N quadrature points (the same points generated by 
	#			self.gaussLegendre).
	#
	# Returns:
	#		L2_k -- relative L2 error in the approximation
	#		L2_k0 -- L2 error of the "exact" function
	#				for normalizing the loss function.
	#------------------------------------------------------------------	
	def computeError(self, u0):

		# compute L2 error
		EXACT = np.reshape(self.phiTrain, [self.Ny*self.Nt, 1])
		L2_k = np.sqrt(np.sum(self.wGlobal * (u0 - EXACT)**2))
		L2_k0 = np.sqrt(np.sum(self.wGlobal * (EXACT)**2))
		L2_k = L2_k/L2_k0

		return L2_k, L2_k0


	#------------------------------------------------------------------
	# Function appendBasis
	#
	# Purpose: append the current basis function to the list self.basis
	#			evaluated at the necessary quadrature points.
	#
	# Parameters:
	#		params -- hidden parameters of the current basis function.
	#		coeffs -- weights of the activation layer of the current
	#				basis function.
	#		scale -- scaling parameter for the activation function. larger
	#				values induce sharper gradients of the activation.
	#
	# Returns:
	#
	#------------------------------------------------------------------	
	def appendBasis(self, params, coeffs, scale):

		tTrain = np.reshape(self.xGlobal[:,1], [self.xGlobal.shape[0],1])
		yTrain = np.reshape(self.xGlobal[:,0], [self.xGlobal.shape[0],1])

		# append phi_i to basis, evaluated at training points
		phi_i = self.createNetwork(tTrain, yTrain, params, coeffs, scale)
		self.basis.append(phi_i)


		tTest = np.reshape(self.xGlobalVal[:,1], [self.xGlobalVal.shape[0],1])
		yTest = np.reshape(self.xGlobalVal[:,0], [self.xGlobalVal.shape[0],1])

		# append phi_i to basis, evaluated at validation points
		phival_i = self.createNetwork(tTest, yTest, params, coeffs, scale)
		self.basis_val.append(phival_i)

		return



	#------------------------------------------------------------------
	# Function galerkinSolve
	#
	# Purpose: solve the problem using the first k basis functions.
	#
	# Parameters:
	#		k -- k+1 is the current number of basis functions computed.
	#
	# Returns:
	#		uTrain -- approximation to the function using first k basis functions
	#				evaluated at interior quadrature points.
	#		uVal -- " " " evaluated at interior validation quad points.
	#		c -- coefficients of each basis function.
	#
	#------------------------------------------------------------------	
	def galerkinSolve(self, k):

		# generate best approximation as linear combination of basis functions
		A = np.zeros([k+1, k+1])
		b = np.zeros([k+1, 1])

		# assemble linear system
		for idx in range(k+1):
			for jdx in range(k+1):
				A[idx,jdx] = np.sum(self.wGlobal * (self.basis[idx]) * (self.basis[jdx]))
			b[idx] = np.sum(self.wGlobal * (self.basis[idx]) * np.reshape(self.phiTrain, [self.Ny*self.Nt,1]))


		# if MATLAB engine is installed, use it
		# c = eng.lsqminnorm(matlab.double(A.tolist()), matlab.double(b.tolist()))
		[c, _, _, _] = scipy.linalg.lstsq(A, b)
		c = np.asarray(c)
		c = np.reshape(c, [k+1,1])

		print("Coefficients of the basis functions")
		print(c)

		# compute linear combination
		uTrain = np.zeros(self.wGlobal.shape)

		uVal = np.zeros(self.wGlobalVal.shape)

		for idx in range(k+1):
			uTrain += c[idx] * self.basis[idx]

			uVal += c[idx] * self.basis_val[idx]

		return uTrain, uVal, c


	#------------------------------------------------------------------
	# Function plotError
	#
	# Purpose: plot exact Riesz representation u-u_{i-1} and basis 
	#		   function \varphi_i^NN
	#
	# Parameters:
	#		k -- k+1 is the current number of basis functions computed.
	#
	# Returns:
	#
	#------------------------------------------------------------------	
	def plotError(self, k, uVal):

		T = np.reshape(self.xGlobalVal[:,1], [self.NyVal, self.NtVal])
		Y = np.reshape(self.xGlobalVal[:,0], [self.NyVal, self.NtVal])
		ERROR = np.reshape(self.phiVal - uVal, [self.NyVal, self.NtVal])
		BASIS = np.reshape(self.basis_val[-1], [self.NyVal, self.NtVal])

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot_surface(T, Y, ERROR, cmap=cm.viridis)
		ax.set_xlabel(r'$t$')
		ax.set_ylabel(r'$y$')
		ax.set_zlabel(r'$u - u_{i-1}$')
		ax.set_title('Error' + ', i=' + str(k+1))
		ax.set_box_aspect((3, 1, 1))
		plt.savefig(self.RESULTS_PATH + "/plot_error_%d" % k)
		plt.close()

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot_surface(T, Y, BASIS, cmap=cm.viridis)
		ax.set_xlabel(r'$t$')
		ax.set_ylabel(r'$y$')
		ax.set_zlabel(r'$\varphi_{i}^{NN}$')
		ax.set_title('Basis Function' + ', i=' + str(k+1))
		ax.set_box_aspect((3, 1, 1))
		plt.savefig(self.RESULTS_PATH + "/plot_basis_%d" % k)
		plt.close()

		return


	#------------------------------------------------------------------
	# Function plotSolution
	#
	# Purpose: plot the approximation u_i as a 2D plot plus one-dimensional
	#		   slices
	#
	# Parameters:
	#		k -- k+1 is the current number of basis functions computed.
	#		uVal -- approximation u_i evaluated at the quadrature nodes
	#
	# Returns:
	#
	#------------------------------------------------------------------	
	def plotSolution(self, k, uVal):

		T = np.reshape(self.xGlobalVal[:,1], [self.NyVal, self.NtVal])
		Y = np.reshape(self.xGlobalVal[:,0], [self.NyVal, self.NtVal])
		Ytrain = np.reshape(self.xGlobal[:,0], [self.Ny, self.Nt])
		PHI = np.reshape(uVal, [self.NyVal, self.NtVal])
		EXACT = np.reshape(self.phiVal, [self.NyVal, self.NtVal])
		EXACTtrain = np.reshape(self.phiTrain, [self.Ny, self.Nt])

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot_surface(T, Y, PHI, cmap=cm.viridis)
		ax.set_xlabel(r'$\gamma$')
		ax.set_ylabel('$y/a$')
		ax.set_zlabel(r'$\phi$')
		ax.set_title('Volume Fraction Approximation')
		ax.set_box_aspect((3, 1, 1))
		plt.savefig(self.RESULTS_PATH + "/plot_solution_%d" % k)
		plt.close()

		plt.figure()
		n = 20
		plt.plot(Y[:,n], EXACT[:,n], label="Exact")
		# plt.plot(Ytrain[:,20], EXACTtrain[:,20], label="Exact train")
		plt.plot(Y[:,n], PHI[:,n], label="Galerkin")
		plt.xlabel(r'$y$')
		plt.ylabel(r'$u$')
		mystr = r'Volume Fraction at $t=$' + str(np.round_(T[0,n])) + ', i=' + str(k+1)
		plt.title(mystr)
		plt.legend()
		plt.grid()
		plt.savefig(self.RESULTS_PATH + "/plot_phi_slice1_%d" % k)
		plt.close()

		plt.figure()
		n = 55
		plt.plot(Y[:,n], EXACT[:,n], label="Exact")
		plt.plot(Y[:,n], PHI[:,n], label="Galerkin")
		plt.xlabel(r'$y$')
		plt.ylabel(r'$u$')
		mystr = r'Volume Fraction at $t=$' + str(np.round_(T[0,n])) + ', i=' + str(k+1)
		plt.title(mystr)
		plt.legend()
		plt.grid()
		plt.savefig(self.RESULTS_PATH + "/plot_phi_slice2_%d" % k)
		plt.close()

		plt.figure()
		n = 85
		plt.plot(Y[:,n], EXACT[:,n], label="Exact")
		plt.plot(Y[:,n], PHI[:,n], label="Galerkin")
		plt.xlabel(r'$y$')
		plt.ylabel(r'$u$')
		mystr = r'Volume Fraction at $t=$' + str(np.round_(T[0,n])) + ', i=' + str(k+1)
		plt.title(mystr)
		plt.legend()
		plt.grid()
		plt.savefig(self.RESULTS_PATH + "/plot_phi_slice3_%d" % k)
		plt.close()

		plt.figure()
		n = 135
		plt.plot(Y[:,n], EXACT[:,n], label="Exact")
		plt.plot(Y[:,n], PHI[:,n], label="Galerkin")
		plt.xlabel(r'$y$')
		plt.ylabel(r'$u$')
		mystr = r'Volume Fraction at $t=$' + str(np.round_(T[0,n])) + ', i=' + str(k+1)
		plt.title(mystr)
		plt.legend()
		plt.grid()
		plt.savefig(self.RESULTS_PATH + "/plot_phi_slice4_%d" % k)
		plt.close()

		return



	# helper function for computing gradients of the loss function
	def update(self, params, coeffs, scalek, T, Y, W, U0, F, opt_state, opt_update, get_params):
	    """ Compute the gradient for a batch and update the parameters """
	    value, grads = jax.value_and_grad(self.computeLoss, argnums=0)(params, T, Y, W, coeffs, scalek, U0, F)
	    opt_state = opt_update(0, grads, opt_state)

	    return get_params(opt_state), opt_state, value


	def generateBasis(self):

		fTrain = np.reshape(self.phiTrain, [self.Ny*self.Nt, 1])
		fVal = np.reshape(self.phiVal, [self.NyVal*self.NtVal, 1])

		N = self.xGlobal.shape[0]
		tTrain = np.reshape(self.xGlobal[:,1], [N,1])
		yTrain = np.reshape(self.xGlobal[:,0], [N,1])

		N = self.xGlobalVal.shape[0]
		tVal = np.reshape(self.xGlobalVal[:,1], [N,1])
		yVal = np.reshape(self.xGlobalVal[:,0], [N,1])

		# initial approximation at training points
		uTrain = np.zeros(self.wGlobal.shape)

		# initial approximation at validation points
		uVal = np.zeros(self.wGlobalVal.shape)

		# generate new basis
		errorIndicator = 1.e3
		k = 0

		while (np.abs(errorIndicator) > self.tol and k < self.maxRef):

			# width and scaling for kth basis function
			neuronsk = int(self.neurons * (2**k))
			scalek = np.minimum(self.scale + 0.25*k, 250.5)
			# scalek = np.minimum(self.scale + 3.*k, 950.0)
			sizesk = [2, neuronsk, neuronsk] #, 6*neuronsk]

			# weights and biases
			params = self.initParams(sizesk)

			# learning rate
			lr0 = (5.e-3) / (1.1**k)
			opt_init, opt_update, get_params = optimizers.adam(lr0)
			opt_state = opt_init(params)

			# Get the initial set of parameters
			params = get_params(opt_state)

			# compute Energy and L2 error
			L2_k, L2_k0 = self.computeError(uTrain)


			gradCheck0 = 1.e5
			gradCheck1 = 1.e2

			i = 0

			# for i in range(NUM_EPOCHS):
			while (np.abs(gradCheck0 - gradCheck1)/np.abs(gradCheck0) > self.gradTol and i < self.maxEpoch):

				# update activation coefficients
				coeffs = self.galerkinUpdate(sizesk[-1], params, uTrain, scalek)

				# evaluate loss and approximate L2
				loss_i = self.computeLoss(params, tTrain, yTrain, self.wGlobal, coeffs, scalek, uTrain, fTrain)
				loss_i = -loss_i / L2_k0

				self.L2_epoch.append(L2_k)
				self.Loss_epoch.append(loss_i)


				print("Iter: ", k, ", Epoch: ", i, ", Loss: ", loss_i, ",  L2: ", L2_k)

				# get current weights and biases
				params, opt_state, loss = self.update(params, coeffs, scalek, tTrain, yTrain, self.wGlobal, 
													  uTrain, fTrain, opt_state, opt_update, get_params)

				gradCheck0 = gradCheck1
				gradCheck1 = loss_i

				i += 1

			# last activation coefficient update
			coeffs = self.galerkinUpdate(sizesk[-1], params, uTrain, scalek)

			# evaluate loss and approximate L2
			loss_i = self.computeLoss(params, tTrain, yTrain, self.wGlobal, coeffs, scalek, uTrain, fTrain)
			loss_i = -loss_i / L2_k0
			
			# append phi_i to basis
			self.appendBasis(params, coeffs, scalek)

			# plot basis function vs. true error
			self.plotError(k, uVal)

			# compute approximation to variational problem using basis functions
			uTrain, uVal, c = self.galerkinSolve(k)

			# save parameters of current basis function
			trainedParams_k = optimizers.unpack_optimizer_state(opt_state)
			self.trainedParams.append([trainedParams_k, coeffs, c])

			
			# plot approximation using k basis functions
			self.plotSolution(k, uVal)

			k += 1

			# update error indicator
			errorIndicator = loss_i

			self.L2_epoch.append(L2_k)
			self.Loss_epoch.append(loss_i)

			self.DOFS_iter.append(neuronsk)
			self.L2_iter.append(L2_k)
			self.Loss_iter.append(loss_i)

		# save results
		pickle.dump(self.trainedParams, open(self.RESULTS_PATH + "/results.pkl", "wb"))


		# plot loss per epoch and iteration
		fig = plt.figure()
		plt.subplot(1,2,1)
		plt.semilogy(np.arange(1,len(self.basis)+1), self.L2_iter, 'o-', label=r'$|||\phi-\phi_{i-1}|||$')
		plt.semilogy(np.arange(1,len(self.basis)+1), np.asarray(self.Loss_iter), 'o--', color='tab:orange', label=r'Loss')
		plt.xlabel('Number of basis functions', fontsize=16)
		plt.ylabel('Error', fontsize=16)
		mystr = r'Error in $\phi$ vs. Number of Basis Functions'
		plt.title(mystr, fontsize=16)
		plt.legend(fontsize=12)
		ratio = 0.75
		plt.grid()

		plt.subplot(1,2,2)
		plt.semilogy(self.L2_epoch, '-', label=r'$|||\phi-\phi_{i-1}|||$')
		plt.semilogy(np.asarray(self.Loss_epoch), '--', color='tab:orange', label=r'Loss')
		plt.xlabel('Epoch', fontsize=16)
		plt.ylabel('Error', fontsize=16)
		mystr = r'Error in $\phi$ vs. Epoch'
		plt.title(mystr, fontsize=16)
		plt.legend(fontsize=12)
		ratio = 0.75
		plt.grid()
		plt.show()




