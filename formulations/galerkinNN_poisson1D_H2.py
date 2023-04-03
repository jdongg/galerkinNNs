import jax.numpy as jnp
from jax.example_libraries import optimizers
import jax
from joblib import Parallel, delayed
import numpy as np

import matplotlib.pyplot as plt
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import scipy.io as sio
from scipy.special import roots_jacobi
import scipy
from functools import reduce

from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

# import matlab.engine
# eng = matlab.engine.start_matlab()

#------------------------------------------------------------------
# class GNN
#
# Purpose: this class creates a Galerkin neural network and uses
#		   it to approximate the solution to a 1D Poisson equation.
#		   the formulation used is the H2 least squares formulation:
#				a(u,v) = (u'',v'') + C1*(u,v)_{Gamma} + C2*(u',v')_{Gamma}
#				L(v) = -(f,v'')
#
# Parameters:
#		neurons -- width of the neural network for the first basis
#				   function. widths for subsequent networks follow
#				   the rule neurons * 20^{i-1}
#		scale -- initial scaling argument for the activation function
#						sigma(scale * t).
#				 subsequent activation functions follow the rule
#				 scale + 2^{i-1}
#		f -- function for RHS data
#		u, ux, uxx -- function for true solution and its derivative
#		a, b -- 1D domain
#		ng -- number of nodes for training
#		tolerance -- for generating basis functions. stoping computing
#					 new basis functions when 
#						<r(u_{i-1}), \varphi_i^NN> < tolerance
#		gradTolerance -- for minimizing loss function. stop when 
#							|loss_i - loss_{i-1}| / |loss_{i-1}| < gradTolerance
#		boundaryPen1, boundaryPen2 -- penalty parameters for boundary terms
#		maxEpoch -- maximum number of iterations to train each 
#					basis function for
#		maxRef -- maximum number of basis functions to compute
#		RESULTS_PATH -- directory for saving results
#
#------------------------------------------------------------------
class GNN:
	def __init__(self, neurons, scale, f, u, ux, uxx, a, b, ng, tolerance, gradTolerance, 
				 boundaryPen1, boundaryPen2, maxEpoch, maxRef, RESULTS_PATH):

		# path for saving results
		self.RESULTS_PATH = RESULTS_PATH

		# problem-specific functions
		self.sourceFunc = f
		self.exactSol = u
		self.exactDx = ux
		self.exactDx2 = uxx

		self.xa = a
		self.xb = b

		# generate quadrature rule in interior; training
		self.xGlobal, self.wGlobal = self.gaussLegendre(a, b, ng)

		# generate quadrature rule in interior; validation
		self.xGlobalVal, self.wGlobalVal = self.gaussLegendre(a, b, 2000)

		# on boundary, we only need to evaluate the functions on the boundary (i.e. wBdry = 1)
		self.xBdry = np.concatenate(([[self.xa]], [[self.xb]]))
		self.wBdry = np.ones([2,1])

		# list of basis functions of their derivatives
		self.basis = []
		self.basisxx = []
		self.basisbdry = []
		self.basisbdryx = []

		self.basis_val = []
		self.basisxx_val = []

		# tolerance for adaptive basis generation
		self.tol = tolerance
		self.gradTol = gradTolerance

		self.boundaryPen1 = boundaryPen1
		self.boundaryPen2 = boundaryPen2

		# initial network width
		self.neurons = neurons
		self.scale = scale

		self.maxEpoch = maxEpoch
		self.maxRef = maxRef

		# results epochs
		self.L2_epoch = []
		self.Loss_epoch = []
		self.Energy_epoch = []

		self.DOFS_iter = []
		self.L2_iter = []
		self.Loss_iter = []
		self.Energy_iter = []


	#------------------------------------------------------------------
	# Function gaussLegendre
	#
	# Purpose: generate a 1D Gauss-Legendre quadrature rule
	#
	# Parameters:
	#		a, b -- 1D domain
	#		ng -- number of nodes to return
	#
	# Returns:
	#		x, w -- nodes and weights for a Gauss-Legendre quadrature rule
	#				in the interval (a,b)
	#------------------------------------------------------------------
	def gaussLegendre(self, a, b, ng):

		x, w = roots_jacobi(ng, 0, 0) 
		x = np.reshape(x, [ng,1])
		w = np.reshape(w, [ng,1])

		x = 0.5*(b-a)*x + 0.5*(b+a)
		w = 0.5*(b-a)*w

		return x, w


	#------------------------------------------------------------------
	# Function initLayer
	#
	# Purpose: initialize hidden parameters for a hidden layer
	#
	# Parameters:
	#		m, n -- input and output dimension of the layer
	#		xa, xb -- 1D domain
	#		L -- total number of hidden layers
	#		ell -- current layer number
	#
	# Returns:
	#		W, b -- weights and biases for hidden layer ell
	#------------------------------------------------------------------
	def initLayer(self, m, n, xa, xb, L, ell):

		# first hidden layer
		if (m == 1):
			# hidden weights and biases
			W =  np.ones([m, n])
			b = np.linspace(xa, xb, n)
			b =  -np.reshape(b, [1, n])


			# # box initialization (Cyr, et. al. 2020)
			# p = np.array(np.random.uniform(size=[n, m]))
			# norm = np.array(np.random.normal(size=[n, m]))

			# nhat = np.zeros([n, m])
			# pmax = np.zeros([n, m])
			# k = np.zeros([1, n])

			# W = np.zeros([n, m])
			# b = np.zeros([n, 1])

			# for i in range(n):
			# 	for j in range(m):
			# 		nhat[i,j] = norm[i,j] / np.linalg.norm(norm[i,:], 2)
			# 		pmax[i,j] = np.maximum(-1.0, np.sign(nhat[i,j]))

			# for i in range(n):
			# 	k[0,i] = 1.0 / np.abs(1.0 * np.sum((pmax[i,:] - p[i,:]) * nhat[i,:]))
			# 	# print(k[i])
			# 	for j in range(m):
			# 		W[i,j] = k[0,i]*nhat[i,j]

			# 	b[i] = -k[0,i]*np.sum(nhat[i,:] * p[i,:])

			# Wr = W.T
			# W = W.T
			# b = b.T
		

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
	# Purpose: initialize hidden parameters for the network
	#
	# Parameters:
	#		sizes -- list of widths for each layer in the network 
	#				 (including input layer)
	#		a, b -- 1D domain
	#
	# Returns:
	#		params -- list of parameters for each hidden layer with the
	#				  structure 
	#						W, b = params[i]
	#------------------------------------------------------------------
	def initParams(self, sizes, a, b):
		return [self.initLayer(m, n, a, b, len(sizes), i) for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:]))]

	
	#------------------------------------------------------------------
	# Function createNetwork
	#
	# Purpose: construct the feedforward neural network
	#
	# Parameters:
	#		X -- array of inputs (must be shape (N,1))
	#		params -- list of hidden parameters as generated by initParams
	#		C -- linear coefficients of the network as generated by 
	#			 galerkinUpdate
	#		scale -- scaling factor for activation function
	#
	# Returns:
	#		U -- array of outputs (shape(U) = shape(X))
	#------------------------------------------------------------------
	@partial(jax.jit, static_argnums=(0,))
	def createNetwork(self, X, params, C, scale):

		# first hidden layer
		W, b = params[0]
		H = jnp.tanh(scale * (X * W + b)) 
		Hend = H

		for W, b in params[1:]:
			H = jnp.tanh(scale * (jnp.matmul(Hend, W) + b)) 
			Hend = H

		U = Hend @ C

		return U


	#------------------------------------------------------------------
	# Function networkArray
	#
	# Purpose: evaluate the individual neurons of the network at X
	#
	# Parameters:
	#		X -- array of inputs (must be shape (N,1))
	#		params -- list of hidden parameters as generated by initParams
	#		scale -- scaling factor for activation function
	#
	# Returns:
	#		Hend -- (N,n) array where n is the width of the last hidden
	#			    hidden layer. Hend[:,i] corresponds to the function
	#						sigma(x*W_i + b_i)
	#------------------------------------------------------------------
	@partial(jax.jit, static_argnums=(0,))
	def networkArray(self, X, params, scale):

		W, b = params[0]
		H = jnp.tanh(scale * (X * W + b)) 
		Hend = H

		for W, b in params[1:]:
			H = jnp.tanh(scale * (jnp.matmul(Hend, W) + b))
			Hend = H

		return Hend


	#------------------------------------------------------------------
	# Function activMatrix
	#
	# Purpose: generate the linear system Kc = F for updating the
	#		   linear coefficients c of the network
	#
	# Parameters:
	#		i -- for loop index for parallelization
	#		NN_dx2 -- Nxn array consisting of second derivatives of 
	#				 each neuron. NN_dx2[:,i] contains
	#					sigma''(x*W_i + b_i)
	#		NNbdry -- 2xn array consisting of boundary values for
	#				  each neuron. NN_bdry[:,i] contains
	#					sigma([a,b].T*W_i) + b_i)
	#		NNbdry -- " derivative boundary values for
	#				  each neuron. NN_bdry[:,i] contains
	#					sigma'([a,b].T*W_i) + b_i)
	#		u0xx -- second derivative of the previous approximation
	#			   u_{i-1}
	#		u0bdry -- boundary values of previous approx. u_{i-1}
	#		u0bdryx -- boundary derivative values of previous approx. u_{i-1}
	#		K, F -- arrays for the linear system to be filled
	#
	# Returns:
	#		in-place modification of K and F
	#------------------------------------------------------------------
	def activMatrix(self, i, NN_dx2, NNbdry, NNbdry_dx, u0xx, u0bdry, u0bdryx, K, F):

		NNdx2_idx = np.reshape(NN_dx2[:,i], self.xGlobal.shape)
		NNdx2_jdx = NN_dx2

		NNbdry_idx = np.reshape(NNbdry[:,i], self.xBdry.shape)
		NNbdry_jdx = NNbdry

		NNbdrydx_idx = np.reshape(NNbdry_dx[:,i], self.xBdry.shape)
		NNbdrydx_jdx = NNbdry_dx

		# equivalent to a(u, v)
		K[i,:] = np.sum(self.wGlobal * (NNdx2_idx) * (NNdx2_jdx), axis=0)
		K[i,:] += self.boundaryPen1 * np.sum(self.wBdry * (NNbdry_idx) * (NNbdry_jdx), axis=0)
		K[i,:] += self.boundaryPen2 * np.sum(self.wBdry * (NNbdrydx_idx) * (NNbdrydx_jdx), axis=0)
	 
	 	# RHS: equivalent to L(v) - a(u0, v)
		F[i] = np.sum(self.wGlobal * (-NNdx2_idx) * (self.sourceFunc(self.xGlobal)))
		F[i] += self.boundaryPen1 * np.sum(self.wBdry * (NNbdry_idx) * (self.exactSol(self.xBdry)))
		F[i] += self.boundaryPen2 * np.sum(self.wBdry * (NNbdrydx_idx) * (self.exactDx(self.xBdry)))
		F[i] -= np.sum(self.wGlobal * (NNdx2_idx) * (u0xx))
		F[i] -= self.boundaryPen1 * np.sum(self.wBdry * (NNbdry_idx) * u0bdry)
		F[i] -= self.boundaryPen2 * np.sum(self.wBdry * (NNbdrydx_idx) * u0bdryx)

		return



	#------------------------------------------------------------------
	# Function galerkinUpdate
	#
	# Purpose: updates linear coefficients of the network given the
	#		   the hidden parameters
	#
	# Parameters:
	#		neurons -- width of the last hidden layer
	#		params -- list of hidden parameters as generated by initParams
	#		u0xx -- second derivative of the previous approximation
	#				u_{i-1}
	#		u0bdry -- boundary values of previous approx. u_{i-1}
	#		u0bdryx -- boundary derivative values of previous approx. u_{i-1}
	#		K, F -- arrays for the linear system to be filled
	#
	# Returns:
	#		in-place modification of K and F
	#------------------------------------------------------------------
	def galerkinUpdate(self, neurons, params, u0xx, u0bdry, u0bdryx, scale):

		N = self.xGlobal.shape[0]
		K = np.zeros([neurons, neurons])
		F = np.zeros([neurons, 1])

		# evaluate network function 
		NNoutput = self.networkArray(self.xGlobal, params, scale)
		NNbdry = self.networkArray(self.xBdry, params, scale)

		# derivative on the boundary
		gradx = jax.vmap(jax.jacfwd(self.networkArray, 0), (0, None, None), 0)
		NNbdry_dx = jnp.squeeze(gradx(self.xBdry, params, scale))

		# evaluate second derivative of network function
		gradxx = jax.vmap(jax.jacfwd(jax.jacfwd(self.networkArray, 0), 0), (0, None, None), 0)
		NN_dx2 = jnp.squeeze(gradxx(self.xGlobal, params, scale))

		# assemble matrices, columns can be done in parallel
		Parallel(n_jobs=8, backend="threading")(
			delayed(self.activMatrix)(i, NN_dx2, NNbdry, NNbdry_dx, u0xx, u0bdry, u0bdryx, K, F) for i in range(neurons))


		# c = eng.lsqminnorm(matlab.double(A.tolist()), matlab.double(b.tolist()))
		c, _, _, _ = scipy.linalg.lstsq(K, F, cond=None)
		# c = np.linalg.solve(K, F)
		c = np.reshape(c, [neurons,1])

		return c


	#------------------------------------------------------------------
	# Function computeLoss
	#
	# Purpose: evaluate the loss function
	#				<r(u_{i-1}), v> / |||v|||
	#
	# Parameters:
	#		params -- list of hidden parameters as generated by initParams
	#		X, W -- quadrature rule in interior of domain
	#		Xbdry, Wbdry -- " on boundary of domain
	#		coeffs -- linear coefficients of network obtained from
	#				  galerkinUpdate
	#		U0xx -- second derivative of the previous approximation u_{i-1}
	#		U0bdry -- boundary values of previous approx. u_{i-1}
	#		U0bdryx -- boundary derivative values of previous approx. u_{i-1}
	#		F -- RHS of PDE
	#		G -- Dirichlet boundary condition
	#
	# Returns:
	#		Loss -- loss function, a scalar
	#------------------------------------------------------------------
	@partial(jax.jit, static_argnums=(0,))
	def computeLoss(self, params, X, W, Xbdry, Wbdry, coeffs, scalek, U0xx, U0bdry, U0bdryx, F, G, Gx):
		
		phibdry = self.createNetwork(Xbdry, params, coeffs, scalek)

		gradx = jax.vmap(jax.jacfwd(self.createNetwork, 0), (0, None, None, None), 0)
		phibdryx = jnp.reshape(gradx(Xbdry, params, coeffs, scalek), U0bdry.shape)

		gradxx = jax.vmap(jax.jacfwd(jax.jacfwd(self.createNetwork, 0), 0), (0, None, None, None), 0)
		phixx = jnp.reshape(gradxx(X, params, coeffs, scalek), U0xx.shape)

		# compute loss function
		normint = jnp.sum(jnp.multiply(W, jnp.square(phixx)))
		normbdry = self.boundaryPen1 * jnp.sum(jnp.multiply(Wbdry, jnp.square(phibdry)))
		normbdryx = self.boundaryPen2 * jnp.sum(jnp.multiply(Wbdry, jnp.square(phibdryx)))
		norm = jnp.sqrt(normint + normbdry + normbdryx)
		

		# a(u0, v) = (u0, v)
		r1 = jnp.sum(jnp.multiply(W, jnp.multiply(U0xx, phixx)))
		r2 = self.boundaryPen1 * jnp.sum(jnp.multiply(Wbdry, jnp.multiply(U0bdry, phibdry)))
		r3 = self.boundaryPen2 * jnp.sum(jnp.multiply(Wbdry, jnp.multiply(U0bdryx, phibdryx)))

		# L(v) = (f, v)
		r4 = jnp.sum(jnp.multiply(W, jnp.multiply(F, -phixx)))
		r5 = self.boundaryPen1 * jnp.sum(jnp.multiply(Wbdry, jnp.multiply(G, phibdry)))
		r6 = self.boundaryPen2 * jnp.sum(jnp.multiply(Wbdry, jnp.multiply(Gx, phibdryx)))

		# L(v) - a(u0, v)
		res = (r4 + r5 + r6) - (r1 + r2 + r3)

		# max L = min -L, where L is 
		#	L(v) - a(u0, v)
		#	---------------
		#	 a(v, v)^(1/2)
		Loss = jnp.multiply(-1.0, jnp.abs(jnp.divide(res, norm)))

		return Loss


	#------------------------------------------------------------------
	# Function computeError
	#
	# Purpose: computes error in the energy and L2 norms
	#
	# Parameters:
	#		u0, u0xx, u0bdry u0bdryx -- relevant values of previous approximation
	#						   u_{i-1}
	#
	# Returns:
	#		Energy_k, L2_k -- normalized energy and L2 error
	#		Energy_k0, L2_k0 -- energy and L2 norms of the true solution
	#							for normalizing the loss function
	#------------------------------------------------------------------
	def computeError(self, u0, u0xx, u0bdry, u0bdryx):

		# compute Energy and L2 error
		Energy_k = np.sum(self.wGlobalVal * (u0xx - self.exactDx2(self.xGlobalVal))**2)
		Energy_k += self.boundaryPen1 * np.sum(self.wBdry * (u0bdry - self.exactSol(self.xBdry))**2)
		Energy_k += self.boundaryPen2 * np.sum(self.wBdry * (u0bdryx - self.exactDx(self.xBdry))**2)
		Energy_k = np.sqrt(Energy_k)

		Energy_k0 = np.sum(self.wGlobalVal * (self.exactDx2(self.xGlobalVal))**2)
		Energy_k0 += self.boundaryPen1 * np.sum(self.wBdry * (self.exactSol(self.xBdry))**2)
		Energy_k0 += self.boundaryPen2 * np.sum(self.wBdry * (self.exactDx(self.xBdry))**2)
		Energy_k0 = np.sqrt(Energy_k0)

		Energy_k = Energy_k / Energy_k0

		L2_k = np.sqrt(np.sum(self.wGlobalVal * (u0 - self.exactSol(self.xGlobalVal))**2))
		L2_k0 = np.sqrt(np.sum(self.wGlobalVal * (self.exactSol(self.xGlobalVal))**2))
		L2_k = L2_k / L2_k0

		return Energy_k, Energy_k0, L2_k, L2_k0


	#------------------------------------------------------------------
	# Function appendBasis
	#
	# Purpose: stores the computed basis function \varphi_i^NN and its
	#		   derivatives
	#
	# Parameters:
	#		params -- hidden parameters for \varphi_i^NN
	#		coeffs -- linear/activation coefficients for \varphi_i^NN
	#		scale -- scaling argument for activation function
	#
	# Returns:
	#
	#------------------------------------------------------------------
	def appendBasis(self, params, coeffs, scale):

		# append phi_i to basis, evaluated at training points
		phi_i = self.createNetwork(self.xGlobal, params, coeffs, scale)
		phibdry_i = self.createNetwork(self.xBdry, params, coeffs, scale)
		
		gradx = jax.vmap(jax.jacfwd(self.createNetwork, 0), (0, None, None, None), 0)
		phibdryx_i = jnp.reshape(gradx(self.xBdry, params, coeffs, scale), phibdry_i.shape)

		gradxx = jax.vmap(jax.jacfwd(jax.jacfwd(self.createNetwork, 0), 0), (0, None, None, None), 0)
		phixx_i = jnp.reshape(gradxx(self.xGlobal, params, coeffs, scale), phi_i.shape)

		self.basis.append(phi_i)
		self.basisxx.append(phixx_i)
		self.basisbdry.append(phibdry_i)
		self.basisbdryx.append(phibdryx_i)

		# append phi_i to basis, evaluated at validation points
		phival_i = self.createNetwork(self.xGlobalVal, params, coeffs, scale)
		phixxval_i = jnp.reshape(gradxx(self.xGlobalVal, params, coeffs, scale), phival_i.shape)

		self.basis_val.append(phival_i)
		self.basisxx_val.append(phixxval_i)

		return


	#------------------------------------------------------------------
	# Function galerkinSolve
	#
	# Purpose: compute linear combination of the basis functions
	#		   varphi_i^NN based on Galerkin method
	#
	# Parameters:
	#		k -- k+1 is the number of basis functions computed so far
	#		fTrain -- RHS of PDE
	#		gTrain, gxTrain -- Dirichlet boundary condition
	#
	# Returns:
	#		_Train -- new approximation evaluated at training
	#				  quadrature points
	#		_Val -- " evaluated at validation quadrature points
	#		c -- coefficients of the linear combination
	#
	#------------------------------------------------------------------
	def galerkinSolve(self, k, fTrain, gTrain, gxTrain):

		# generate best approximation as linear combination of basis functions
		A = np.zeros([k+1, k+1])
		b = np.zeros([k+1, 1])

		# assemble linear system
		for idx in range(k+1):
			for jdx in range(k+1):
				A[idx,jdx] = np.sum(self.wGlobal * (self.basisxx[idx]) * (self.basisxx[jdx]))
				A[idx,jdx] += self.boundaryPen1 * np.sum(self.wBdry * (self.basisbdry[idx]) * (self.basisbdry[jdx]))
				A[idx,jdx] += self.boundaryPen2 * np.sum(self.wBdry * (self.basisbdryx[idx]) * (self.basisbdryx[jdx]))
			b[idx] = np.sum(self.wGlobal * (-self.basisxx[idx]) * (fTrain))
			b[idx] += self.boundaryPen1 * np.sum(self.wBdry * (self.basisbdry[idx]) * gTrain)
			b[idx] += self.boundaryPen2 * np.sum(self.wBdry * (self.basisbdryx[idx]) * gxTrain)


		# if MATLAB engine is installed, use it
		# c = eng.lsqminnorm(matlab.double(A.tolist()), matlab.double(b.tolist()))
		[c, _, _, _] = scipy.linalg.lstsq(A, b)
		c = np.asarray(c)
		c = np.reshape(c, [k+1,1])

		print("Coefficients of linear combination:")
		print(c)

		# compute linear combinations
		uTrain = np.zeros(self.xGlobal.shape)
		uxxTrain = np.zeros(self.xGlobal.shape)
		uBdry = np.zeros(self.xBdry.shape)
		uBdryx = np.zeros(self.xBdry.shape)

		uVal = np.zeros(self.xGlobalVal.shape)
		uxxVal = np.zeros(self.xGlobalVal.shape)

		for idx in range(k+1):
			uTrain += c[idx] * self.basis[idx]
			uxxTrain += c[idx] * self.basisxx[idx]
			uBdry += c[idx] * self.basisbdry[idx]
			uBdryx += c[idx] * self.basisbdryx[idx]

			uVal += c[idx] * self.basis_val[idx]
			uxxVal += c[idx] * self.basisxx_val[idx]

		return uTrain, uxxTrain, uBdry, uBdryx, uVal, uxxVal, c


	#------------------------------------------------------------------
	# Function plotError
	#
	# Purpose: plot the true Riesz representation and basis function
	#
	# Parameters:
	#		k -- basis function counter
	#		uVal-- solution u_i at validation points
	#
	# Returns:
	#
	#------------------------------------------------------------------
	def plotError(self, k, uVal):

		plt.figure()
		plt.plot(self.xGlobalVal, (self.exactSol(self.xGlobalVal) - uVal), label=r'Exact $u-u_{i-1}$')
		plt.plot(self.xGlobalVal, self.basis_val[-1], '--', label=r'$\varphi_{i}^{NN}$')
		plt.xlabel('x', fontsize=16)
		plt.ylabel('y', fontsize=16)
		mystr = 'Exact and Predicted Error in $u_{i-1}$, i=' + str(k+1)
		plt.title(mystr, fontsize=16)
		plt.legend(fontsize=16)
		plt.grid()
		plt.savefig(self.RESULTS_PATH + "/plot_error_%d" % k)
		plt.close()

		return


	#------------------------------------------------------------------
	# Function plotSolution
	#
	# Purpose: plot approximate solution u_i
	#
	# Parameters:
	#		k -- basis function counter
	#		uVal-- solution u_i at validation points
	#
	# Returns:
	#
	#------------------------------------------------------------------
	def plotSolution(self, k, uVal):

		plt.figure()
		plt.plot(self.xGlobalVal, self.exactSol(self.xGlobalVal), label='Exact')
		plt.plot(self.xGlobalVal, uVal, '--', label='Predicted')
		plt.xlabel('$x$', fontsize=16)
		plt.ylabel('$y$', fontsize=16)
		mystr = 'Exact and Predicted Solution, i=' + str(k+1)
		plt.title(mystr, fontsize=16)
		plt.legend(fontsize=16)
		plt.grid()
		plt.savefig(self.RESULTS_PATH + "/plot_solution_%d" % k)
		plt.close()

		return


	#------------------------------------------------------------------
	# Function L2estimator
	#
	# Purpose: computes L2 norm of the basis function as an estimator for
	#		   the L2 error ||u-u_{i-1}||
	#
	# Parameters:
	#		X, W -- quadrature rule in interior of domain
	#		params -- hidden parameters of basis function
	#		coeffs -- linear/activation coefficients of basis function
	#		scalek -- scaling arg for activation function
	#
	# Returns:
	#		L2 -- L2 norm of the basis function \varphi_i^NN
	#------------------------------------------------------------------
	def L2estimator(self, X, W, params, coeffs, scalek):
		phi = self.createNetwork(X, params, coeffs, scalek)

		L2 = jnp.sqrt(jnp.sum(jnp.multiply(W, jnp.square(phi))))

		return L2


	def update(self, params, coeffs, scalek, U0xx, U0bdry, U0bdryx, F, G, Gx, opt_state, opt_update, get_params):
	    """ Compute the gradient for a batch and update the parameters """
	    value, grads = jax.value_and_grad(self.computeLoss, argnums=0)(params, self.xGlobal, self.wGlobal,
	    															   self.xBdry, self.wBdry, coeffs, scalek, 
	    															   U0xx, U0bdry, U0bdryx, F, G, Gx)
	    opt_state = opt_update(0, grads, opt_state)

	    return get_params(opt_state), opt_state, value


	def generateBasis(self):

		fTrain = self.sourceFunc(self.xGlobal)
		fVal = self.sourceFunc(self.xGlobalVal)

		gTrain = self.exactSol(self.xBdry)
		gxTrain = self.exactDx(self.xBdry)

		# initial approximation at training points
		uTrain = np.zeros(self.xGlobal.shape)
		uxxTrain = np.zeros(self.xGlobal.shape)
		uBdry = np.zeros(self.xBdry.shape)
		uBdryx = np.zeros(self.xBdry.shape)

		# initial approximation at validation points
		uVal = np.zeros(self.xGlobalVal.shape)
		uxxVal = np.zeros(self.xGlobalVal.shape)

		# generate new basis
		errorIndicator = 1.e3
		k = 0

		while (np.abs(errorIndicator) > self.tol and k < self.maxRef):
		# while (k < self.maxRef):

			# width and scaling for kth basis function
			neuronsk = int(self.neurons * (2**k))
			scalek = np.minimum(self.scale + 2.**k, 950.0)
			# scalek = np.minimum(self.scale + 3.*k, 950.0)
			sizesk = [1, 2+neuronsk] #, 6*neuronsk]

			# weights and biases
			params = self.initParams(sizesk, self.xa, self.xb)

			# learning rate
			lr0 = (5.e-3) / (1.1**k)
			opt_init, opt_update, get_params = optimizers.adam(lr0)
			opt_state = opt_init(params)

			# Get the initial set of parameters
			params = get_params(opt_state)

			# compute Energy and L2 error
			Energy_k, Energy_k0, L2_k, L2_k0 = self.computeError(uVal, uxxVal, uBdry, uBdryx)


			gradCheck0 = 1.e5
			gradCheck1 = 1.e2

			i = 0

			# for i in range(NUM_EPOCHS):
			while (np.abs(gradCheck0 - gradCheck1)/np.abs(gradCheck0) > self.gradTol and i < self.maxEpoch):

				# update activation coefficients
				coeffs = self.galerkinUpdate(sizesk[-1], params, uxxTrain, uBdry, uBdryx, scalek)
				# print(coeffs)

				# evaluate loss and approximate L2
				loss_i = self.computeLoss(params, self.xGlobalVal, self.wGlobalVal, self.xBdry, self.wBdry,
										  coeffs, scalek, uxxVal, uBdry, uBdryx, fVal, gTrain, gxTrain)
				loss_i = -loss_i / Energy_k0

				self.L2_epoch.append(L2_k)
				self.Loss_epoch.append(loss_i)
				self.Energy_epoch.append(Energy_k)


				print("Iter: ", k, ", Epoch: ", i, ", Loss: ", loss_i, ",  Energy error: ", Energy_k, ", L2 error: ", L2_k)

				# get current weights and biases
				params, opt_state, loss = self.update(params, coeffs, scalek, uxxTrain, uBdry, uBdryx,
													  fTrain, gTrain, gxTrain, opt_state, opt_update, get_params)

				gradCheck0 = gradCheck1
				gradCheck1 = loss_i

				i += 1

			# last activation coefficient update
			coeffs = self.galerkinUpdate(sizesk[-1], params, uxxTrain, uBdry, uBdryx, scalek)


			# evaluate loss and approximate L2
			loss_i = self.computeLoss(params, self.xGlobalVal, self.wGlobalVal, self.xBdry, self.wBdry,
									  coeffs, scalek, uxxVal, uBdry, uBdryx, fVal, gTrain, gxTrain)
			loss_i = -loss_i / Energy_k0
			
			# append phi_i to basis
			self.appendBasis(params, coeffs, scalek)

			# plot basis function vs. true error
			self.plotError(k, uVal)

			

			# compute approximation to variational problem using basis functions
			uTrain, uxxTrain, uBdry, uBdryx, uVal, uxxVal, c = self.galerkinSolve(k, fTrain, gTrain, gxTrain)

			# plot approximation using k basis functions
			self.plotSolution(k, uVal)

			k += 1

			# update error indicator
			errorIndicator = loss_i

			self.L2_epoch.append(L2_k)
			self.Loss_epoch.append(loss_i)
			self.Energy_epoch.append(Energy_k)

			self.DOFS_iter.append(neuronsk)
			self.L2_iter.append(L2_k)
			self.Energy_iter.append(Energy_k)
			self.Loss_iter.append(loss_i)



		# plot loss per epoch and iteration
		fig = plt.figure()
		plt.subplot(1,2,1)
		plt.semilogy(np.arange(1,len(self.basis)+1), self.Energy_iter, 'o-', label=r'$|||u-u_{i-1}|||$')
		plt.semilogy(np.arange(1,len(self.basis)+1), np.asarray(self.Loss_iter), 'o--', color='tab:orange', label=r'$\eta(u_{i-1},\varphi_{i}^{NN})$')
		plt.xlabel('Number of basis functions', fontsize=16)
		plt.ylabel('Error', fontsize=16)
		mystr = r'Error vs. Number of Basis Functions'
		plt.title(mystr, fontsize=16)
		plt.legend(fontsize=12)
		ratio = 0.75
		plt.grid()

		plt.subplot(1,2,2)
		plt.semilogy(self.Energy_epoch, '-', label=r'$|||u-u_{i-1}|||$')
		plt.semilogy(np.asarray(self.Loss_epoch), '--', color='tab:orange', label=r'$\eta(u_{i-1},\varphi_{i}^{NN})$')
		plt.xlabel('Number of basis functions', fontsize=16)
		plt.ylabel('Error', fontsize=16)
		mystr = r'Error vs. Number of Basis Functions'
		plt.title(mystr, fontsize=16)
		plt.legend(fontsize=12)
		ratio = 0.75
		plt.grid()
		plt.show()

		plt.show()
