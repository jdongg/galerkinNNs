import jax.numpy as jnp
from jax.example_libraries import optimizers
import jax
import time
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import scipy
import pickle
import random
from functools import partial
jax.config.update("jax_enable_x64", True)
from typing import Callable, List, Type

from matplotlib import ticker
niceMathTextForm = ticker.ScalarFormatter(useMathText=True)

import sys
sys.path.insert(0, '../QuadratureRules')
from QuadratureRules import QuadratureRules

# exact value of leading eigenvalue for Stokes eigenproblem with non-convex angle
# pi + arccos(1/sqrt(10))
alphaFixedU = 1.58223874

class GNN:
    """
    The GNN class is responsible for learning a sequence of basis functions which form a subspace for solving
    the variational problem

                u ∈ X : a(u,v) = L(v)    ∀v ∈ X.

    The basic assumptions on the operators as as follows:

        * L is continuous
        * a is continuous and symmetric positive-definite.

    For problems which do not admit a naturally-SPD operator a, the least squares variational formulation is used. 
    Given L : X -> V, B : X -> W and data f ∈ V, g ∈ W, the boundary value problem 

                L[u] = f    in Ω
                B[u] = g    on ∂Ω

    corresponds to the least squares variational problem given by

                a(u,v) := (L[u], L[v])_V + C*(B[u], B[v])_W
                L(v) := (f, L[v])_V + C*(g, B[v])_W

    which is always SPD. This class solves the stationary Stokes equation using a least-sqaures variational formulation posed
    on H2 x H1:

                a((u,p),(v,q)) = (-L(u) + grad(p), -L(v) + grad(q))_L2(Ω) + (div(u), div(v))_H1(Ω) + C*(u,v)_H1(Ω) + <compatibility condition>
                L((v,q))       = (f, -L(v) + grad(q))_L2(Ω) + (g, div(v))_H1(Ω) + C*(u_D, v)_H1(∂Ω) + <compatibility condition>.

    The compatibility condition consists of specifying the pressure value at a single point x0 and takes the form p(x0) * q(x0).
    The class is tailored to solving the Stokes equation in the triangular wedge. The singular functions for the extended NN 
    architecture are set up so that the eigenvalue of the eddy functions is a trainable parameter. This corresponds to 
    Example 4.3 in [1].

    [1] Ainsworth, M., & Dong, J. (2024). Extended Galerkin neural network approximation of singular variational 
        problems with error control. arXiv preprint arXiv:2405.00815.

    The user is expected to provide the following information:

        * neurons         : width of hidden layers for the first basis function
        * scale           : 
        * Lx, Ly          : length and width of bounding box around Ω; used to determine scaling for Fourier feature mapping
        * isExtended      : bool specifying whether extended NN architecture is used
        * QuadTrain       : quadrature rules for integrating over the domain and its boundary; includes tangential and normal
                            vectors for computing normal and tangential derivatives and vector components
        * QuadVal         : quadrature rule for computing errors; must be different from QuadTrain to ensure network isn't
                            overfit
        * xStream         : points at which the solution is to be plotted; assumed to have shape N^2 for some integer N
        * f               : function handle for RHS data f
        * u               : function handle for evaluating exact solution on boundary and for L2 errors (when closed form available)
        * Du              : function handle for gradient of exact solution on boundary
        * tolerance       : tolerance for determining how many basis functions to compute, i.e. after computing a basis function, if
                            loss < tolerance, stop. otherwise, generate new basis function.
        * gradTolerance   : tolerance for determining when to stop training each basis function, i.e. if relative change in loss < gradTolerance,
                            stop training.
        * boundaryPen1    : penalty parameter for L2 inner products on boundary
        * boundaryPen2    : penalty parameter for H1 seminorm inner products on boundary
        * scale_increment :
        * base_lr         : learning rate for first basis function
        * iter_decay_rate : decay rate for learning rate wrt basis function, i.e. ith basis has learning rate base_lr / iter_decay_rate^i
        * exp_decay_rate  : decay rate for exponential learning schedule
        * maxEpoch        : maximum epochs for training each basis function
        * maxRef          : maximum number of basis functions to compute
        * RESULTS_PATH    : directory for saving figures, parameters, etc.
        * beta            : weight for weighted Sobolev spaces, i.e. r^beta.

    When GNN is initialized and the generateBasis routine is called, up to maxRef basis functions are learned and their
    parameters are saved.
    """
    def __init__(self, neurons, scale, isExtended: bool, Lx: np.float64, Ly: np.float64, 
                 QuadTrain: Type[QuadratureRules], QuadVal: Type[QuadratureRules], xStream,
                 f, u, Du, tolerance, gradTolerance, 
                 boundaryPen1, boundaryPen2, boundaryPen3, H1DivergenceFlag,
                 scale_increment, base_lr, iter_decay_rate, exp_decay_rate,  
                 maxEpoch, maxRef, RESULTS_PATH, beta):

        # path for saving results
        self.RESULTS_PATH = RESULTS_PATH

        self.SCALE_INCR = scale_increment
        self.BASE_LR = base_lr 
        self.ITER_DECAY = iter_decay_rate
        self.EXP_DECAY = exp_decay_rate

        # problem-specific functions
        self.sourceFunc = f
        self.exactSol = u
        self.exactDx = Du

        self.boundaryPen1 = boundaryPen1
        self.boundaryPen2 = boundaryPen2
        self.boundaryPen3 = boundaryPen3
        self.H1flag = H1DivergenceFlag

        # width of bounding rectangle for domain
        self.Lx = Lx
        self.Ly = Ly 

        # weighted formulation power
        self.beta = beta

        # generate quadrature rule; training
        self.QuadRules = QuadTrain

        # generate quadrature rule; validation
        self.QuadRulesVal = QuadVal
        
        self.xStream0 = xStream

        self.isExtendedNetwork = isExtended
        self.alphaSize = 1

        # cutoff function parameters. the cutoff function takes the value 1 for r < r0, 
        # 0 for r > r1, and is C2 for r0 <= r <= r1. the coefficients held in C are the 
        # coefficients of the quintic polynomial, i.e.
        #       c0 * r^5 + c1 * c^4 + c2 * r^3 + c4 * r^2 + c5 * r + c6.
        self.r0 = 2.7
        self.r1 = 2.8

        A = [[self.r0**5, self.r0**4, self.r0**3, self.r0**2, self.r0, 1],
                [self.r1**5, self.r1**4, self.r1**3, self.r1**2, self.r1, 1],
                [5*self.r0**4, 4*self.r0**3, 3*self.r0**2, 2*self.r0, 1, 0],
                [5*self.r1**4, 4*self.r1**3, 3*self.r1**2, 2*self.r1, 1, 0],
                [20*self.r0**3, 12*self.r0**2, 6*self.r0, 2, 0, 0],
                [20*self.r1**3, 12*self.r1**2, 6*self.r1, 2, 0, 0]]
        b = [[1], [0], [0], [0], [0], [0]]

        self.C = np.squeeze(scipy.linalg.solve(A, b))

        # list of solution fields
        self.solution_fields = ["u1", "u2", "p"]

        # list of derivatives in the interior, spatial boundary, and temporal boundary
        # for each solution field
        #   0 - x
        #   1 - y
        # 
        self.intDerivatives = {}
        self.bdryDerivatives = {}
        self.tbdryDerivatives = {}

        self.intDerivatives["u1"] = ["0", "00", "11", "01"] # for u1
        self.intDerivatives["u2"] = ["1", "00", "11", "10"] # for u2
        self.intDerivatives["p"] = ["0", "1"] # for p
        self.bdryDerivatives["u1"] = ["0", "1"] # for u1
        self.bdryDerivatives["u2"] = ["0", "1"] # for u2
        self.bdryDerivatives["p"] = []
        self.tbdryDerivatives["u1"] = []
        self.tbdryDerivatives["u2"] = []
        self.tbdryDerivatives["p"] = []

        # specify which solution fields have spatial boundary
        self.hasSpatialBoundary = {}
        self.hasSpatialBoundary["u1"] = True
        self.hasSpatialBoundary["u2"] = True
        self.hasSpatialBoundary["p"] = False

        self.hasCompatibilityCondition = {}
        self.hasCompatibilityCondition["u1"] = False
        self.hasCompatibilityCondition["u2"] = False
        self.hasCompatibilityCondition["p"] = True

        # list of basis functions of their derivatives; these are lists of dictionaries, e.g.
        # self.basis["u1"][i] returns a dictionary for the various derivatives and value of phi1.
        # e.g. self.basis["u1"]["xx"] returns the second x derivative of the ith basis function for u1
        self.basis = {}
        self.basis_val = {}
        self.basis_stream = {}

        for field in self.solution_fields:
            self.basis[field] = []
            self.basis_val[field] = []
            self.basis_stream[field] = []

        # tolerance for adaptive basis generation
        self.tol = tolerance
        self.gradTol = gradTolerance

        # initial network width
        self.neurons = neurons
        self.scale = scale

        self.maxEpoch = maxEpoch
        self.maxRef = maxRef

        # flag for marking when to switch to singular basis
        self.iterFlag = 0

        # results epochs
        self.L2_epoch = []
        self.Loss_epoch = []
        self.approxL2_epoch = []
        self.Energy_epoch = []
        self.theta_epoch = []
        self.zeta_epoch = []
        self.xi_epoch = []

        self.L2_iter = []
        self.L2p_iter = []
        self.approxL2_iter = []
        self.L2weighted_iter = []
        self.approxL2weighted_iter = []
        self.Energy_iter = []
        self.natEnergy_iter = []
        self.Loss_iter = []
        self.cond_iter = []

        # list which contains the trained parameters for each basis function
        self.trainedParams = []


    def parseDerivativeString(self, der):
        der_print = ""
        for der_i in der:
            if (der_i == "0"):
                der_print += "x"
            else:
                der_print += "y"
        return der_print 

        
    def boxInit(self, m, n, flag, ki, L, ell, numWaves):
        """
        Initializes a hidden layer according to the Cyr, et. al. box initilization algorithm.

        Cyr, E. C., Gulian, M. A., Patel, R. G., Perego, M., & Trask, N. A. (2020, August). 
        Robust training and initialization of deep neural networks: An adaptive basis viewpoint. 
        In Mathematical and Scientific Machine Learning (pp. 512-536). PMLR.
        """
        if (m == 2 * self.QuadRules.interior_x.shape[1] * numWaves):
            p = np.array(np.random.uniform(size=[n, m]))
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

            W = W.T
            b = b.T
            Wr = np.ones([m, n])
        else:
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
                for j in range(m):
                    W[i,j] = k[0,i]*nhat[i,j]

                b[i] = -k[0,i]*np.sum(nhat[i,:] * p[i,:])

            Wr = W.T
            W = W.T
            b = b.T

        return W, b, Wr


    def initLayer(self, m, n, L, ell, k, numWaves):

        if (n == 1):
            Wphi1 = np.zeros((m, n))
            Wphi2 = np.zeros((m, n))
            Wq = np.zeros((m, n))

            return Wphi1, Wphi2, Wq
        else:
            Wphi1, bphi1, Wrphi1 = self.boxInit(m, n, 0, k, L, ell, numWaves,)
            Wphi2, bphi2, Wrphi2 = self.boxInit(m, n, 1, k, L, ell, numWaves)
            Wq, bq, Wrq = self.boxInit(m, n, 2, k, L, ell, numWaves)

            alphaOffset1 = random.uniform(-0.7, 0.7)
            alphaOffset2 = random.uniform(-0.7, 0.7)

            alpha = (np.pi - 2.0*np.arccos(1.0/np.sqrt(10.0))) / 2.0
            zetaFixedU = 1.0 + 4.2266/(2.0*alpha)
            xiFixedU = 2.17466/(2.0*alpha)

            zeta = np.reshape(np.linspace(zetaFixedU, 1.047176+1.0, 1), [1, 1]) + alphaOffset1
            xi = np.reshape(np.linspace(xiFixedU, 1.047176+1.0, 1), [1, 1]) + alphaOffset2

            scalephi1 = self.scale + self.SCALE_INCR * k
            scalephi2 = self.scale + self.SCALE_INCR * k
            scaleq = self.scale + self.SCALE_INCR * k

            m = jnp.arange(0, numWaves)
            wxm = jnp.pow(2.0, m) * jnp.pi / self.Lx
            wym = jnp.pow(2.0, m) * jnp.pi / self.Ly

            return (Wphi1, bphi1, Wrphi1, zeta, xi, scalephi1, wxm, wym,
                    Wphi2, bphi2, Wrphi2, zeta, xi, scalephi2, wxm, wym,
                    Wq, bq, Wrq, zeta, xi, scaleq, wxm, wym)
        return


    def initParams(self, sizes, k, numWaves):
        return [self.initLayer(m, n, len(sizes)-1, i+1, k, numWaves) for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:]))]


    def separateParams(self, params):
        paramsphi1 = []
        paramsphi2 = []
        paramsq = []

        for i in range(len(params)):
            (Wphi1, bphi1, Wrphi1, zeta, xi, scalephi1, wxmphi1, wymphi1,
             Wphi2, bphi2, Wrphi2, _, _, scalephi2, wxmphi2, wymphi2, 
             Wq, bq, Wrq, _, _, scaleq, wxmq, wymq) = params[i]

            paramsphi1.append((Wphi1, bphi1, Wrphi1, zeta, xi, scalephi1, wxmphi1, wymphi1))
            paramsphi2.append((Wphi2, bphi2, Wrphi2, zeta, xi, scalephi2, wxmphi2, wymphi2))
            paramsq.append((Wq, bq, Wrq, zeta, xi, scaleq, wxmq, wymq))

        params_i = {}
        params_i["u1"] = paramsphi1
        params_i["u2"] = paramsphi2
        params_i["p"] = paramsq

        return params_i
    

    @partial(jax.jit, static_argnums=(0,6,))
    def createNetwork(self, X, Y, params, Cnn, scale, field, Calpha=None):
        """
        Forward pass for creating feedforward DNN with Fourier feature mapping of input layer [1,2].
        An optional extended architecture is used [3] which supplements the DNN with singular functions
        based on eigenfunctions of the corresponding operator pencil. There are three sets of singular
        functions for this problem corresponding to the two corners at (-1,0) and (1,0) and the eddies
        originating from (0,-3). See [3] for more details. 

        [1] Aldirany, Z., Cottereau, R., Laforest, M., & Prudhomme, S. (2024). Multi-level neural networks 
            for accurate solutions of boundary-value problems. Computer Methods in Applied Mechanics and Engineering, 419, 116666.
        [2] Tancik, M., Srinivasan, P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., ... & Ng, R. (2020). 
            Fourier features let networks learn high frequency functions in low dimensional domains. Advances in neural 
            information processing systems, 33, 7537-7547.
        [3] Ainsworth, M., & Dong, J. (2024). Extended Galerkin neural network approximation of singular variational 
            problems with error control. arXiv preprint arXiv:2405.00815.
        """
        # cutoff function
        R = jnp.sqrt(jnp.square(X) + jnp.square(Y+3.0))
        T = jnp.arctan2(Y+3.0, X) - np.pi/2.0
        R = R.reshape(-1, 1)
        T = T.reshape(-1, 1)
        conds = [R < self.r0, (R > self.r0) & (R < self.r1), (R > self.r1)]
        funcs = [lambda R: R/R, lambda R: self.C[0]*(R**5) + self.C[1]*(R**4) + self.C[2]*(R**3) + self.C[3]*(R**2) + self.C[4]*R + self.C[5], lambda R: R*0.0]
        cutoff = (1.e0) * jnp.piecewise(R, conds, funcs)

        # first hidden layer
        W, b, Wr, A, B, scale_NN, wxm, wym = params[0]

        Z = jnp.column_stack((jnp.cos(jnp.outer(X, wxm)), jnp.sin(jnp.outer(X, wxm)), 
                                jnp.cos(jnp.outer(Y, wym)), jnp.sin(jnp.outer(Y, wym))))
        H = jnp.tanh((jnp.matmul(Z, W) + b))

        for W, b, Wr, _, _, scale_NN, _, _ in params[1:]:
            H = jnp.tanh((jnp.matmul(H, W) + b)) #+ jnp.matmul(Hend, Wr)

        H = H * (1.0 - cutoff)
        U = H @ Cnn

        if (self.isExtendedNetwork):
            alpha = np.reshape(np.linspace(alphaFixedU, alphaFixedU+1, self.alphaSize), [1,self.alphaSize])
            X = X.reshape(-1, 1)
            Y = Y.reshape(-1, 1)

            # compute re-usable quantities for eddy functions
            conds = [R < self.r0, (R > self.r0) & (R < self.r1), (R > self.r1)]
            funcs = [lambda R: R/R, lambda R: self.C[0]*(R**5) + self.C[1]*(R**4) + self.C[2]*(R**3) + self.C[3]*(R**2) + self.C[4]*R + self.C[5], lambda R: R*0.0]
            cutoff = (1.e0) * jnp.piecewise(R, conds, funcs)

            alpha0 = (np.pi - 2.0*np.arccos(1.0/np.sqrt(10.0))) / 2.0
            # A = 1.0 + 4.2266 / (2.0 * alpha0)
            # B = 2.17466 / (2.0 * alpha0)

            R = jnp.sqrt(jnp.square(X) + jnp.square(Y+3.0))
            T = jnp.arctan2(Y+3.0, X) - np.pi/2.0

            dRdx = (X) / R
            dRdy = (Y+3.0) / R
            dTdx = -(Y+3.0) / (R**2)
            dTdy = (X) / (R**2)

            ReR = (R**A) * jnp.cos(jnp.log(R)*B);
            ImR = (R**A) * jnp.sin(jnp.log(R)*B);

            dReR = A*(R**(A-1)) * jnp.cos(jnp.log(R)*B) + (R**A) * -jnp.sin(jnp.log(R)*B) * (B/R);
            dImR = A*(R**(A-1)) * jnp.sin(jnp.log(R)*B) + (R**A) * jnp.cos(jnp.log(R)*B) * (B/R);

            ReT = (jnp.cos(A*alpha0) * jnp.cos(T*(A-2)) * jnp.cosh(B*alpha0) * jnp.cosh(T*B) - 
                    jnp.sin(A*alpha0) * jnp.sin(T*(A-2)) * jnp.sinh(B*alpha0) * jnp.sinh(T*B) + 
                -(jnp.cos((A-2)*alpha0) * jnp.cos(T*A) * jnp.cosh(B*alpha0) * jnp.cosh(T*B) - 
                    jnp.sin((A-2)*alpha0) * jnp.sin(T*A) * jnp.sinh(B*alpha0) * jnp.sinh(T*B)))

            ImT = (-jnp.cos(A*alpha0) * jnp.sin(T*(A-2)) * jnp.cosh(B*alpha0) * jnp.sinh(T*B) + 
                    -jnp.sin(A*alpha0) * jnp.cos(T*(A-2)) * jnp.sinh(B*alpha0) * jnp.cosh(T*B) +
                -(-jnp.cos((A-2)*alpha0) * jnp.sin(T*A) * jnp.cosh(B*alpha0) * jnp.sinh(T*B) +
                    -jnp.sin((A-2)*alpha0) * jnp.cos(T*A) * jnp.sinh(B*alpha0) * jnp.cosh(T*B)))

            dReT = (jnp.cos(A*alpha0)*jnp.cosh(B*alpha0) * (-(A-2)*jnp.sin(T*(A-2)) * jnp.cosh(T*B) + B*jnp.cos(T*(A-2)) * jnp.sinh(T*B)) -
                jnp.sin(A*alpha0)*jnp.sinh(B*alpha0) * ((A-2)*jnp.cos(T*(A-2)) * jnp.sinh(T*B) + B*jnp.sin(T*(A-2)) * jnp.cosh(T*B)) +
                -(jnp.cos((A-2)*alpha0)*jnp.cosh(B*alpha0) * (-A*jnp.sin(T*A) * jnp.cosh(T*B) + B*jnp.cos(T*A) * jnp.sinh(T*B)) -
                jnp.sin((A-2)*alpha0)*jnp.sinh(B*alpha0) * (A*jnp.cos(T*A) * jnp.sinh(T*B) + B*jnp.sin(T*A) * jnp.cosh(T*B))))

            dImT = (-jnp.cos(A*alpha0)*jnp.cosh(B*alpha0) * ((A-2)*jnp.cos(T*(A-2)) * jnp.sinh(T*B) + B*jnp.sin(T*(A-2)) * jnp.cosh(T*B)) +
                -jnp.sin(A*alpha0)*jnp.sinh(B*alpha0) * (-(A-2)*jnp.sin(T*(A-2)) * jnp.cosh(T*B) + B*jnp.cos(T*(A-2)) * jnp.sinh(T*B)) +
                -(-jnp.cos((A-2)*alpha0)*jnp.cosh(B*alpha0) * (A*jnp.cos(T*A) * jnp.sinh(T*B) + B*jnp.sin(T*A) * jnp.cosh(T*B)) +
                -jnp.sin((A-2)*alpha0)*jnp.sinh(B*alpha0) * (-A*jnp.sin(T*A) * jnp.cosh(T*B) + B*jnp.cos(T*A) * jnp.sinh(T*B))))

            if (field == "u1"):
                # left non-convex corner
                R = jnp.sqrt(jnp.square(X-1.0) + jnp.square(Y))
                offset = (np.pi + np.arccos(1.0/np.sqrt(10.0))) / 2.0
                T = jnp.mod(jnp.arctan2(Y, X-1.0) + 2.0*np.pi, 2.0*np.pi) - offset
                
                R = R.reshape(-1, 1)
                T = T.reshape(-1, 1)

                lambda0 = (np.pi + np.arccos(1.0/np.sqrt(10.0))) / 2.0
                psiR = jnp.power(R, alpha - 2.0)
                psiT = (jnp.multiply(jnp.multiply(Y, alpha * jnp.cos(alpha * lambda0)), jnp.cos(jnp.multiply(T, alpha-2.0))) +
                        -jnp.multiply(jnp.multiply(X - 1.0, (alpha - 2.0) * jnp.cos(alpha * lambda0)), jnp.sin(jnp.multiply(T, alpha-2.0))) +
                        -jnp.multiply(jnp.multiply(Y, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.cos(jnp.multiply(T, alpha))) + 
                        jnp.multiply(jnp.multiply(X - 1.0, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.sin(jnp.multiply(T, alpha))))
                Ualpha = jnp.multiply(psiR, psiT) * (1.0 - cutoff)
                U += Ualpha @ Calpha["leftcorner"][field]


                # right non-convex corner
                R = jnp.sqrt(jnp.square(X+1.0) + jnp.square(Y))
                offset = (np.pi - np.arccos(1.0/np.sqrt(10.0))) / 2.0
                T = jnp.arctan2(Y, X+1.0) - offset

                R = R.reshape(-1, 1)
                T = T.reshape(-1, 1)

                psiR = jnp.power(R, alpha - 2.0)
                psiT = (jnp.multiply(jnp.multiply(Y, alpha * jnp.cos(alpha * lambda0)), jnp.cos(jnp.multiply(T, alpha-2.0))) +
                        -jnp.multiply(jnp.multiply(X + 1.0, (alpha - 2.0) * jnp.cos(alpha * lambda0)), jnp.sin(jnp.multiply(T, alpha-2.0))) +
                        -jnp.multiply(jnp.multiply(-Y, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.cos(jnp.multiply(T, alpha))) + 
                        jnp.multiply(jnp.multiply(X + 1.0, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.sin(jnp.multiply(T, alpha))))
                Ualpha = jnp.multiply(psiR, psiT) * (1.0 - cutoff)
                U += Ualpha @ Calpha["rightcorner"][field]


                # Moffatt eddies
                H1 = (dReR * ReT * dRdy + ReR * dReT * dTdy - (dImR * ImT * dRdy + ImR * dImT * dTdy)) 
                H2 = (dReR * ImT * dRdy + ReR * dImT * dTdy + dImR * ReT * dRdy + ImR * dReT * dTdy) 
                Ualpha = jnp.column_stack((H1, H2)) * cutoff
                U += Ualpha @ Calpha["eddies"][field]

            elif (field == "u2"):
                # left non-convex corner
                R = jnp.sqrt(jnp.square(X-1.0) + jnp.square(Y))
                offset = (np.pi + np.arccos(1.0/np.sqrt(10.0))) / 2.0
                T = jnp.mod(jnp.arctan2(Y, X-1.0) + 2.0*np.pi, 2.0*np.pi) - offset
                
                R = R.reshape(-1, 1)
                T = T.reshape(-1, 1)

                lambda0 = (np.pi + np.arccos(1.0/np.sqrt(10.0))) / 2.0
                psiR = jnp.power(R, alpha - 2.0)
                psiT = (-jnp.multiply(jnp.multiply(X - 1.0, alpha * jnp.cos(alpha * lambda0)), jnp.cos(jnp.multiply(T, alpha-2.0))) +
                        -jnp.multiply(jnp.multiply(Y, (alpha - 2.0) * jnp.cos(alpha * lambda0)), jnp.sin(jnp.multiply(T, alpha-2.0))) +
                         jnp.multiply(jnp.multiply(X - 1.0, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.cos(jnp.multiply(T, alpha))) + 
                         jnp.multiply(jnp.multiply(Y, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.sin(jnp.multiply(T, alpha))))
                Ualpha = jnp.multiply(psiR, psiT) * (1.0 - cutoff)
                U += Ualpha @ Calpha["leftcorner"][field]


                # right non-convex corner
                R = jnp.sqrt(jnp.square(X+1.0) + jnp.square(Y))
                offset = (np.pi - np.arccos(1.0/np.sqrt(10.0))) / 2.0
                T = jnp.arctan2(Y, X+1.0) - offset

                R = R.reshape(-1, 1)
                T = T.reshape(-1, 1)

                # Ualpha = (R**(alpha-2.0)) * (-jnp.cos(alpha*lambda0)*(alpha*(X+1.0)*jnp.cos((alpha-2.0)*T) + (alpha-2.0)*Y*jnp.sin((alpha-2.0)*T)) +
                # 							  alpha*jnp.cos((alpha-2.0)*lambda0)*((X+1.0)*jnp.cos(alpha*T) + Y*jnp.sin(alpha*T)))
                psiR = jnp.power(R, alpha - 2.0)
                psiT = (-jnp.multiply(jnp.multiply(X + 1.0, alpha * jnp.cos(alpha * lambda0)), jnp.cos(jnp.multiply(T, alpha-2.0))) +
                        -jnp.multiply(jnp.multiply(Y, (alpha - 2.0) * jnp.cos(alpha * lambda0)), jnp.sin(jnp.multiply(T, alpha-2.0))) +
                         jnp.multiply(jnp.multiply(X + 1.0, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.cos(jnp.multiply(T, alpha))) + 
                         jnp.multiply(jnp.multiply(Y, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.sin(jnp.multiply(T, alpha))))
                Ualpha = jnp.multiply(psiR, psiT) * (1.0 - cutoff)
                U += Ualpha @ Calpha["rightcorner"][field]


                # Moffatt eddies
                H1 = (-(dReR * ReT * dRdx + ReR * dReT * dTdx) + (dImR * ImT * dRdx + ImR * dImT * dTdx))
                H2 = (-(dReR * ImT * dRdx + ReR * dImT * dTdx) - (dImR * ReT * dRdx + ImR * dReT * dTdx))
                Ualpha = jnp.column_stack((H1, H2)) * cutoff
                U += Ualpha @ Calpha["eddies"][field]

            elif (field == "p"):
                # left non-convex corner
                R = jnp.sqrt(jnp.square(X-1.0) + jnp.square(Y))
                offset = (np.pi + np.arccos(1.0/np.sqrt(10.0))) / 2.0
                T = jnp.mod(jnp.arctan2(Y, X-1.0) + 2.0*np.pi, 2.0*np.pi) - offset
                
                R = R.reshape(-1, 1)
                T = T.reshape(-1, 1)

                lambda0 = (np.pi + np.arccos(1.0/np.sqrt(10.0))) / 2.0 
                # Ualpha = -4.0*(alpha-1.0)*(R**(alpha-2.0)) * jnp.cos(alpha*lambda0) * jnp.sin((alpha-2.0)*T)
                psiR = jnp.power(R, alpha - 2.0)
                psiT = jnp.multiply(-4.0 * (alpha - 1.0) * jnp.cos(alpha * lambda0), jnp.sin(jnp.multiply(T, alpha - 2.0)))
                Ualpha = jnp.multiply(psiR, psiT) * (1.0 - cutoff)
                U += Ualpha @ Calpha["leftcorner"][field]


                # right non-convex corner
                R = jnp.sqrt(jnp.square(X+1.0) + jnp.square(Y))
                offset = (np.pi - np.arccos(1.0/np.sqrt(10.0))) / 2.0
                T = jnp.arctan2(Y, X+1.0) - offset

                R = R.reshape(-1, 1)
                T = T.reshape(-1, 1)

                # Ualpha = -4.0*(alpha-1.0)*(R**(alpha-2.0)) * jnp.cos(alpha*lambda0) * jnp.sin((alpha-2.0)*T)
                psiR = jnp.power(R, alpha - 2.0)
                psiT = jnp.multiply(-4.0 * (alpha - 1.0) * jnp.cos(alpha * lambda0), jnp.sin(jnp.multiply(T, alpha - 2.0)))
                Ualpha = jnp.multiply(psiR, psiT) * (1.0 - cutoff)
                U += Ualpha @ Calpha["rightcorner"][field]


                # Moffatt eddies
                R = jnp.sqrt(jnp.square(X) + jnp.square(Y+3.0))
                # T = jnp.mod(jnp.arctan2(Y, X-1.0) + 2.0*np.pi, 2.0*np.pi)
                T = jnp.arctan2(Y+3.0, X) - np.pi/2.0

                R = R.reshape(-1, 1)
                T = T.reshape(-1, 1)

                H1 = -4.0*(R**(A-2.0)) * (jnp.cos(A*alpha0)*jnp.cosh(B*alpha0) * (jnp.cos(T*(A-2.0))*jnp.sinh(T*B) * (-(A-1.0)*jnp.sin(jnp.log(R)*B) - B*jnp.cos(jnp.log(R)*B)) - jnp.sin(T*(A-2.0))*jnp.cosh(T*B) * (B*jnp.sin(jnp.log(R)*B) - (A-1.0)*jnp.cos(jnp.log(R)*B))) + 
                                          jnp.sin(A*alpha0)*jnp.sinh(B*alpha0) * (jnp.cos(T*(A-2.0))*jnp.sinh(T*B) * ((A-1.0)*jnp.cos(jnp.log(R)*B) - B*jnp.sin(jnp.log(R)*B)) + jnp.sin(T*(A-2.0))*jnp.cosh(T*B) * ((A-1.0)*jnp.sin(jnp.log(R)*B) + B*jnp.cos(jnp.log(R)*B))) )
                H2 = -4.0*(R**(A-2.0)) * (jnp.sin(A*alpha0)*jnp.sinh(B*alpha0) * (jnp.cos(T*(A-2.0))*jnp.sinh(T*B) * ((A-1.0)*jnp.sin(jnp.log(R)*B) + B*jnp.cos(jnp.log(R)*B)) + jnp.sin(T*(A-2.0))*jnp.cosh(T*B) * (B*jnp.sin(jnp.log(R)*B) - (A-1.0)*jnp.cos(jnp.log(R)*B))) + 
                                          jnp.cos(A*alpha0)*jnp.cosh(B*alpha0) * (jnp.cos(T*(A-2.0))*jnp.sinh(T*B) * ((A-1.0)*jnp.cos(jnp.log(R)*B) - B*jnp.sin(jnp.log(R)*B)) + jnp.sin(T*(A-2.0))*jnp.cosh(T*B) * ((A-1.0)*jnp.sin(jnp.log(R)*B) + B*jnp.cos(jnp.log(R)*B))) )
                Ualpha = jnp.column_stack((-H1, -H2)) * cutoff
                U += Ualpha @ Calpha["eddies"][field]

            return jnp.squeeze(U)
        else:
            return jnp.squeeze(U)

        return 


    @partial(jax.jit, static_argnums=(0,5,))
    def networkArrayReg(self, X, Y, params, scale, field):
        """
        Function which returns an N x n array where the jth column represents jth
        component of the last hidden layer (consisting of n neurons) at N quadrature nodes.
        """
        # cutoff function
        R = jnp.sqrt(jnp.square(X) + jnp.square(Y+3.0))
        T = jnp.arctan2(Y+3.0, X) - np.pi/2.0
        R = R.reshape(-1, 1)
        T = T.reshape(-1, 1)
        conds = [R < self.r0, (R > self.r0) & (R < self.r1), (R > self.r1)]
        funcs = [lambda R: R/R, lambda R: self.C[0]*(R**5) + self.C[1]*(R**4) + self.C[2]*(R**3) + self.C[3]*(R**2) + self.C[4]*R + self.C[5], lambda R: R*0.0]
        cutoff = (1.e0) * jnp.piecewise(R, conds, funcs)

        W, b, Wr, A, B, scale_NN, wxm, wym = params[0]

        Z = jnp.column_stack((jnp.cos(jnp.outer(X, wxm)), jnp.sin(jnp.outer(X, wxm)), 
                                jnp.cos(jnp.outer(Y, wym)), jnp.sin(jnp.outer(Y, wym))))
        Hnn = jnp.tanh((jnp.matmul(Z, W) + b))

        for W, b, Wr, _, _, scale_NN, _, _ in params[1:]:
            Hnn = jnp.tanh((jnp.matmul(Hnn, W) + b)) #+ jnp.matmul(Hend, Wr)

        Hnn = Hnn * (1.0 - cutoff)

        if (self.isExtendedNetwork):
            alpha = np.reshape(np.linspace(alphaFixedU, alphaFixedU+1, self.alphaSize), [1,self.alphaSize])
            X = X.reshape(-1, 1)
            Y = Y.reshape(-1, 1)

            # compute re-usable quantities for eddy functions
            conds = [R < self.r0, (R > self.r0) & (R < self.r1), (R > self.r1)]
            funcs = [lambda R: R/R, lambda R: self.C[0]*(R**5) + self.C[1]*(R**4) + self.C[2]*(R**3) + self.C[3]*(R**2) + self.C[4]*R + self.C[5], lambda R: R*0.0]
            cutoff = (1.e0) * jnp.piecewise(R, conds, funcs)

            alpha0 = (np.pi - 2.0*np.arccos(1.0/np.sqrt(10.0))) / 2.0
            # A = 1.0 + 4.2266 / (2.0 * alpha0)
            # B = 2.17466 / (2.0 * alpha0)

            R = jnp.sqrt(jnp.square(X) + jnp.square(Y+3.0))
            T = jnp.arctan2(Y+3.0, X) - np.pi/2.0

            dRdx = (X) / R
            dRdy = (Y+3.0) / R
            dTdx = -(Y+3.0) / (R**2)
            dTdy = (X) / (R**2)

            ReR = (R**A) * jnp.cos(jnp.log(R)*B);
            ImR = (R**A) * jnp.sin(jnp.log(R)*B);

            dReR = A*(R**(A-1)) * jnp.cos(jnp.log(R)*B) + (R**A) * -jnp.sin(jnp.log(R)*B) * (B/R);
            dImR = A*(R**(A-1)) * jnp.sin(jnp.log(R)*B) + (R**A) * jnp.cos(jnp.log(R)*B) * (B/R);

            ReT = (jnp.cos(A*alpha0) * jnp.cos(T*(A-2)) * jnp.cosh(B*alpha0) * jnp.cosh(T*B) - 
                    jnp.sin(A*alpha0) * jnp.sin(T*(A-2)) * jnp.sinh(B*alpha0) * jnp.sinh(T*B) + 
                -(jnp.cos((A-2)*alpha0) * jnp.cos(T*A) * jnp.cosh(B*alpha0) * jnp.cosh(T*B) - 
                    jnp.sin((A-2)*alpha0) * jnp.sin(T*A) * jnp.sinh(B*alpha0) * jnp.sinh(T*B)))

            ImT = (-jnp.cos(A*alpha0) * jnp.sin(T*(A-2)) * jnp.cosh(B*alpha0) * jnp.sinh(T*B) + 
                    -jnp.sin(A*alpha0) * jnp.cos(T*(A-2)) * jnp.sinh(B*alpha0) * jnp.cosh(T*B) +
                -(-jnp.cos((A-2)*alpha0) * jnp.sin(T*A) * jnp.cosh(B*alpha0) * jnp.sinh(T*B) +
                    -jnp.sin((A-2)*alpha0) * jnp.cos(T*A) * jnp.sinh(B*alpha0) * jnp.cosh(T*B)))

            dReT = (jnp.cos(A*alpha0)*jnp.cosh(B*alpha0) * (-(A-2)*jnp.sin(T*(A-2)) * jnp.cosh(T*B) + B*jnp.cos(T*(A-2)) * jnp.sinh(T*B)) -
                jnp.sin(A*alpha0)*jnp.sinh(B*alpha0) * ((A-2)*jnp.cos(T*(A-2)) * jnp.sinh(T*B) + B*jnp.sin(T*(A-2)) * jnp.cosh(T*B)) +
                -(jnp.cos((A-2)*alpha0)*jnp.cosh(B*alpha0) * (-A*jnp.sin(T*A) * jnp.cosh(T*B) + B*jnp.cos(T*A) * jnp.sinh(T*B)) -
                jnp.sin((A-2)*alpha0)*jnp.sinh(B*alpha0) * (A*jnp.cos(T*A) * jnp.sinh(T*B) + B*jnp.sin(T*A) * jnp.cosh(T*B))))

            dImT = (-jnp.cos(A*alpha0)*jnp.cosh(B*alpha0) * ((A-2)*jnp.cos(T*(A-2)) * jnp.sinh(T*B) + B*jnp.sin(T*(A-2)) * jnp.cosh(T*B)) +
                -jnp.sin(A*alpha0)*jnp.sinh(B*alpha0) * (-(A-2)*jnp.sin(T*(A-2)) * jnp.cosh(T*B) + B*jnp.cos(T*(A-2)) * jnp.sinh(T*B)) +
                -(-jnp.cos((A-2)*alpha0)*jnp.cosh(B*alpha0) * (A*jnp.cos(T*A) * jnp.sinh(T*B) + B*jnp.sin(T*A) * jnp.cosh(T*B)) +
                -jnp.sin((A-2)*alpha0)*jnp.sinh(B*alpha0) * (-A*jnp.sin(T*A) * jnp.cosh(T*B) + B*jnp.cos(T*A) * jnp.sinh(T*B))))
            
            if (field == "u1"):
                # left non-convex corner
                R = jnp.sqrt(jnp.square(X-1.0) + jnp.square(Y))
                offset = (np.pi + np.arccos(1.0/np.sqrt(10.0))) / 2.0
                T = jnp.mod(jnp.arctan2(Y, X-1.0) + 2.0*np.pi, 2.0*np.pi) - offset
                
                R = R.reshape(-1, 1)
                T = T.reshape(-1, 1)

                lambda0 = (np.pi + np.arccos(1.0/np.sqrt(10.0))) / 2.0 
                psiR = jnp.power(R, alpha - 2.0)
                psiT = (jnp.multiply(jnp.multiply(Y, alpha * jnp.cos(alpha * lambda0)), jnp.cos(jnp.multiply(T, alpha-2.0))) +
                        -jnp.multiply(jnp.multiply(X - 1.0, (alpha - 2.0) * jnp.cos(alpha * lambda0)), jnp.sin(jnp.multiply(T, alpha-2.0))) +
                        -jnp.multiply(jnp.multiply(Y, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.cos(jnp.multiply(T, alpha))) + 
                        jnp.multiply(jnp.multiply(X - 1.0, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.sin(jnp.multiply(T, alpha))))
                Ualpha1 = jnp.multiply(psiR, psiT) * (1.0 - cutoff)

                # right non-convex corner
                R = jnp.sqrt(jnp.square(X+1.0) + jnp.square(Y))
                offset = (np.pi - np.arccos(1.0/np.sqrt(10.0))) / 2.0
                T = jnp.arctan2(Y, X+1.0) - offset

                R = R.reshape(-1, 1)
                T = T.reshape(-1, 1)

                psiR = jnp.power(R, alpha - 2.0)
                psiT = (jnp.multiply(jnp.multiply(Y, alpha * jnp.cos(alpha * lambda0)), jnp.cos(jnp.multiply(T, alpha-2.0))) +
                        -jnp.multiply(jnp.multiply(X + 1.0, (alpha - 2.0) * jnp.cos(alpha * lambda0)), jnp.sin(jnp.multiply(T, alpha-2.0))) +
                        -jnp.multiply(jnp.multiply(-Y, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.cos(jnp.multiply(T, alpha))) + 
                        jnp.multiply(jnp.multiply(X + 1.0, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.sin(jnp.multiply(T, alpha))))
                Ualpha2 = jnp.multiply(psiR, psiT) * (1.0 - cutoff)

                # Moffatt eddies
                H1 = (dReR * ReT * dRdy + ReR * dReT * dTdy - (dImR * ImT * dRdy + ImR * dImT * dTdy)) 
                H2 = (dReR * ImT * dRdy + ReR * dImT * dTdy + dImR * ReT * dRdy + ImR * dReT * dTdy) 
                Ualpha3 = jnp.column_stack((H1, H2)) * cutoff

            elif (field == "u2"):
                # left non-convex corner
                R = jnp.sqrt(jnp.square(X-1.0) + jnp.square(Y))
                offset = (np.pi + np.arccos(1.0/np.sqrt(10.0))) / 2.0
                T = jnp.mod(jnp.arctan2(Y, X-1.0) + 2.0*np.pi, 2.0*np.pi) - offset
                
                R = R.reshape(-1, 1)
                T = T.reshape(-1, 1)

                lambda0 = (np.pi + np.arccos(1.0/np.sqrt(10.0))) / 2.0 
                psiR = jnp.power(R, alpha - 2.0)
                psiT = (-jnp.multiply(jnp.multiply(X - 1.0, alpha * jnp.cos(alpha * lambda0)), jnp.cos(jnp.multiply(T, alpha-2.0))) +
                        -jnp.multiply(jnp.multiply(Y, (alpha - 2.0) * jnp.cos(alpha * lambda0)), jnp.sin(jnp.multiply(T, alpha-2.0))) +
                         jnp.multiply(jnp.multiply(X - 1.0, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.cos(jnp.multiply(T, alpha))) + 
                         jnp.multiply(jnp.multiply(Y, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.sin(jnp.multiply(T, alpha))))
                Ualpha1 = jnp.multiply(psiR, psiT) * (1.0 - cutoff)

                # right non-convex corner
                R = jnp.sqrt(jnp.square(X+1.0) + jnp.square(Y))
                offset = (np.pi - np.arccos(1.0/np.sqrt(10.0))) / 2.0
                T = jnp.arctan2(Y, X+1.0) - offset

                R = R.reshape(-1, 1)
                T = T.reshape(-1, 1)

                psiR = jnp.power(R, alpha - 2.0)
                psiT = (-jnp.multiply(jnp.multiply(X + 1.0, alpha * jnp.cos(alpha * lambda0)), jnp.cos(jnp.multiply(T, alpha-2.0))) +
                        -jnp.multiply(jnp.multiply(Y, (alpha - 2.0) * jnp.cos(alpha * lambda0)), jnp.sin(jnp.multiply(T, alpha-2.0))) +
                         jnp.multiply(jnp.multiply(X + 1.0, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.cos(jnp.multiply(T, alpha))) + 
                         jnp.multiply(jnp.multiply(Y, alpha * jnp.cos((alpha-2.0) * lambda0)), jnp.sin(jnp.multiply(T, alpha))))
                Ualpha2 = jnp.multiply(psiR, psiT) * (1.0 - cutoff)

                # Moffatt eddies
                H1 = (-(dReR * ReT * dRdx + ReR * dReT * dTdx) + (dImR * ImT * dRdx + ImR * dImT * dTdx))
                H2 = (-(dReR * ImT * dRdx + ReR * dImT * dTdx) - (dImR * ReT * dRdx + ImR * dReT * dTdx))
                Ualpha3 = jnp.column_stack((H1, H2)) * cutoff

            elif (field == "p"):
                # left non-convex corner
                R = jnp.sqrt(jnp.square(X-1.0) + jnp.square(Y))
                offset = (np.pi + np.arccos(1.0/np.sqrt(10.0))) / 2.0
                T = jnp.mod(jnp.arctan2(Y, X-1.0) + 2.0*np.pi, 2.0*np.pi) - offset
                
                R = R.reshape(-1, 1)
                T = T.reshape(-1, 1)

                lambda0 = (np.pi + np.arccos(1.0/np.sqrt(10.0))) / 2.0 
                psiR = jnp.power(R, alpha - 2.0)
                psiT = jnp.multiply(-4.0 * (alpha - 1.0) * jnp.cos(alpha * lambda0), jnp.sin(jnp.multiply(T, alpha - 2.0)))
                Ualpha1 = jnp.multiply(psiR, psiT) * (1.0 - cutoff)

                # right non-convex corner
                R = jnp.sqrt(jnp.square(X+1.0) + jnp.square(Y))
                offset = (np.pi - np.arccos(1.0/np.sqrt(10.0))) / 2.0
                T = jnp.arctan2(Y, X+1.0) - offset

                R = R.reshape(-1, 1)
                T = T.reshape(-1, 1)

                psiR = jnp.power(R, alpha - 2.0)
                psiT = jnp.multiply(-4.0 * (alpha - 1.0) * jnp.cos(alpha * lambda0), jnp.sin(jnp.multiply(T, alpha - 2.0)))
                Ualpha2 = jnp.multiply(psiR, psiT) * (1.0 - cutoff)

                # Moffatt eddies
                R = jnp.sqrt(jnp.square(X) + jnp.square(Y+3.0))
                T = jnp.arctan2(Y+3.0, X) - np.pi/2.0

                R = R.reshape(-1, 1)
                T = T.reshape(-1, 1)

                H1 = -4.0*(R**(A-2.0)) * (jnp.cos(A*alpha0)*jnp.cosh(B*alpha0) * (jnp.cos(T*(A-2.0))*jnp.sinh(T*B) * (-(A-1.0)*jnp.sin(jnp.log(R)*B) - B*jnp.cos(jnp.log(R)*B)) - jnp.sin(T*(A-2.0))*jnp.cosh(T*B) * (B*jnp.sin(jnp.log(R)*B) - (A-1.0)*jnp.cos(jnp.log(R)*B))) + 
                                          jnp.sin(A*alpha0)*jnp.sinh(B*alpha0) * (jnp.cos(T*(A-2.0))*jnp.sinh(T*B) * ((A-1.0)*jnp.cos(jnp.log(R)*B) - B*jnp.sin(jnp.log(R)*B)) + jnp.sin(T*(A-2.0))*jnp.cosh(T*B) * ((A-1.0)*jnp.sin(jnp.log(R)*B) + B*jnp.cos(jnp.log(R)*B))) )
                H2 = -4.0*(R**(A-2.0)) * (jnp.sin(A*alpha0)*jnp.sinh(B*alpha0) * (jnp.cos(T*(A-2.0))*jnp.sinh(T*B) * ((A-1.0)*jnp.sin(jnp.log(R)*B) + B*jnp.cos(jnp.log(R)*B)) + jnp.sin(T*(A-2.0))*jnp.cosh(T*B) * (B*jnp.sin(jnp.log(R)*B) - (A-1.0)*jnp.cos(jnp.log(R)*B))) + 
                                          jnp.cos(A*alpha0)*jnp.cosh(B*alpha0) * (jnp.cos(T*(A-2.0))*jnp.sinh(T*B) * ((A-1.0)*jnp.cos(jnp.log(R)*B) - B*jnp.sin(jnp.log(R)*B)) + jnp.sin(T*(A-2.0))*jnp.cosh(T*B) * ((A-1.0)*jnp.sin(jnp.log(R)*B) + B*jnp.cos(jnp.log(R)*B))) )
                Ualpha3 = jnp.column_stack((-H1, -H2)) * cutoff

            Hnn = jnp.column_stack((Hnn, Ualpha1, Ualpha2, Ualpha3))
        return Hnn


    def activMatrix(self, i, k, NN, solution0,
                     A11, A12, A22, B1, B2, Fphi1, Fphi2,
                     DATA):
        """
        Routine for assembling the linear system corresponding to 

                u ∈ Φ : a(u,v) = L(v) - a(u0,v)    ∀v ∈ Φ,      (*)

        where Φ = span{Ψ_i} and Ψ_i (i=1,...,n) are the n components of the last hidden layer of the DNN
        (plus potential singular functions if using the extended architecture). The discrete problem (*)
        performs a projection of the error u-u0 onto the space Φ and is a least squares training routine for
        the linear coefficients of the activation layer. More details can be found in [1], which is based
        on the strong-form least squares training introduced in [2].

        For Stokes flow, the variational problem can be expressed as

                (u,p) Φ^2 x Φ : a(u,v) + b(v,p) + b(u,q) + c(p,q) = F(v) + G(q)
                                a(u,v) = (-L(u), -L(v))_L2β(Ω) + (div(u), div(v))_H1β(Ω)
                                            + C*(u,v)_H1β(∂Ω)
                                b(v,p) = (-L(v), grad(p))_L2β(Ω)
                                c(p,q) = (grad(p), grad(q))_L2β(Ω)
                                F(v)   = (f, -L(v))_L2β(Ω) + (g, div(v))_H1β(Ω) + C*(u_D, v)_H1β(∂Ω)
                                G(q)   = (f, grad(q))_L2β(Ω).
        
        The subroutine activMatrix evaluates a(u,v), b(v,p), and F(v) while activMatrix2 evaluates c(p,q)
        and G(q). 

        [1] Ainsworth, M., & Dong, J. (2021). Galerkin neural networks: A framework for approximating 
            variational equations with error control. SIAM Journal on Scientific Computing, 43(4), A2474-A2501.
        [2] Cyr, E. C., Gulian, M. A., Patel, R. G., Perego, M., & Trask, N. A. (2020, August). 
            Robust training and initialization of deep neural networks: An adaptive basis viewpoint. 
            In Mathematical and Scientific Machine Learning (pp. 512-536). PMLR.
        """
        xTrain = np.reshape(self.QuadRules.interior_x[:,0], [self.QuadRules.interior_x.shape[0], 1])
        yTrain = np.reshape(self.QuadRules.interior_x[:,1], [self.QuadRules.interior_x.shape[0], 1])

        xBdry = np.reshape(self.QuadRules.boundary_x[:,0], [self.QuadRules.boundary_x.shape[0], 1])
        yBdry = np.reshape(self.QuadRules.boundary_x[:,1], [self.QuadRules.boundary_x.shape[0], 1])

        if (self.isExtendedNetwork):
            w1 = np.sqrt((xTrain-1.0)**2 + (yTrain)**2)**self.beta
            w2 = np.sqrt((xTrain+1.0)**2 + (yTrain)**2)**self.beta
            w = w1 * w2
            w0 = w

            w1 = np.sqrt((xBdry-1.0)**2 + (yBdry)**2)**self.beta
            w2 = np.sqrt((xBdry+1.0)**2 + (yBdry)**2)**self.beta
            wBdry = w1 * w2
        else:
            w = 1.0
            w0 = 1.0
            wBdry = 1.0

        # interior terms
        # A matrix
        NNdx2phi1_idx = np.reshape(NN["u1"]["interior_xx"][:,i], [self.QuadRules.interior_x.shape[0], 1])
        NNdx2phi1_jdx = NN["u1"]["interior_xx"]

        NNdxyphi1_idx = np.reshape(NN["u1"]["interior_xy"][:,i], [self.QuadRules.interior_x.shape[0], 1])
        NNdxyphi1_jdx = NN["u1"]["interior_xy"]

        NNdy2phi1_idx = np.reshape(NN["u1"]["interior_yy"][:,i], [self.QuadRules.interior_x.shape[0], 1])
        NNdy2phi1_jdx = NN["u1"]["interior_yy"]

        A11[i,:] = np.sum(self.QuadRules.interior_w * (w*(-NNdx2phi1_idx - NNdy2phi1_idx)) * (w*(-NNdx2phi1_jdx - NNdy2phi1_jdx)), axis=0)

        NNdxphi1_idx = np.reshape(NN["u1"]["interior_x"][:,i], [self.QuadRules.interior_x.shape[0], 1])
        NNdxphi1_jdx = NN["u1"]["interior_x"]

        A11[i,:] += np.sum(self.QuadRules.interior_w * (w0*NNdxphi1_idx) * (w0*NNdxphi1_jdx), axis=0)
        A11[i,:] += self.H1flag * np.sum(self.QuadRules.interior_w * (w*NNdx2phi1_idx) * (w*NNdx2phi1_jdx), axis=0)
        A11[i,:] += self.H1flag * np.sum(self.QuadRules.interior_w * (w*NNdxyphi1_idx) * (w*NNdxyphi1_jdx), axis=0)


        NNdx2phi2_idx = np.reshape(NN["u2"]["interior_xx"][:,i], [self.QuadRules.interior_x.shape[0], 1])
        NNdx2phi2_jdx = NN["u2"]["interior_xx"]

        NNdyxphi2_idx = np.reshape(NN["u2"]["interior_yx"][:,i], [self.QuadRules.interior_x.shape[0], 1])
        NNdyxphi2_jdx = NN["u2"]["interior_yx"]

        NNdy2phi2_idx = np.reshape(NN["u2"]["interior_yy"][:,i], [self.QuadRules.interior_x.shape[0], 1])
        NNdy2phi2_jdx = NN["u2"]["interior_yy"]

        A22[i,:] = np.sum(self.QuadRules.interior_w * (w*(-NNdx2phi2_idx - NNdy2phi2_idx)) * (w*(-NNdx2phi2_jdx - NNdy2phi2_jdx)), axis=0)
        
        NNdyphi2_idx = np.reshape(NN["u2"]["interior_y"][:,i], [self.QuadRules.interior_x.shape[0], 1])
        NNdyphi2_jdx = NN["u2"]["interior_y"]

        A22[i,:] += np.sum(self.QuadRules.interior_w * (w0*NNdyphi2_idx) * (w0*NNdyphi2_jdx), axis=0)
        A22[i,:] += self.H1flag * np.sum(self.QuadRules.interior_w * (w*NNdyxphi2_idx) * (w*NNdyxphi2_jdx), axis=0)
        A22[i,:] += self.H1flag * np.sum(self.QuadRules.interior_w * (w*NNdy2phi2_idx) * (w*NNdy2phi2_jdx), axis=0)

        A12[i,:] = np.sum(self.QuadRules.interior_w * (w0*NNdxphi1_idx) * (w0*NNdyphi2_jdx), axis=0)
        A12[i,:] += self.H1flag * np.sum(self.QuadRules.interior_w * (w*NNdx2phi1_idx) * (w*NNdyxphi2_jdx), axis=0)
        A12[i,:] += self.H1flag * np.sum(self.QuadRules.interior_w * (w*NNdxyphi1_idx) * (w*NNdy2phi2_jdx), axis=0)


        # B matrix
        NNdxq_idx = np.reshape(NN["p"]["interior_x"][:,i], [self.QuadRules.interior_x.shape[0], 1])
        NNdxq_jdx = NN["p"]["interior_x"]

        NNdyq_idx = np.reshape(NN["p"]["interior_y"][:,i], [self.QuadRules.interior_x.shape[0], 1])
        NNdyq_jdx = NN["p"]["interior_y"]

        B1[i,:] = np.sum(self.QuadRules.interior_w * (w*(-NNdx2phi1_idx - NNdy2phi1_idx)) * (w*NNdxq_jdx), axis=0)
        B2[i,:] = np.sum(self.QuadRules.interior_w * (w*(-NNdx2phi2_idx - NNdy2phi2_idx)) * (w*NNdyq_jdx), axis=0)


        # boundary terms
        NNbdryphi1_idx = np.reshape(NN["u1"]["boundary_value"][:,i], [self.QuadRules.boundary_x.shape[0], 1])
        NNbdryphi1_jdx = NN["u1"]["boundary_value"]

        NNbdryphi1dx_idx = np.reshape(NN["u1"]["boundary_x"][:,i], [self.QuadRules.boundary_x.shape[0], 1])
        NNbdryphi1dx_jdx = NN["u1"]["boundary_x"]

        NNbdryphi1dy_idx = np.reshape(NN["u1"]["boundary_y"][:,i], [self.QuadRules.boundary_x.shape[0], 1])
        NNbdryphi1dy_jdx = NN["u1"]["boundary_y"]

        A11[i,:] += self.boundaryPen1 * np.sum(self.QuadRules.boundary_w * (wBdry*NNbdryphi1_idx) * (wBdry*NNbdryphi1_jdx), axis=0)
        A11[i,:] += self.boundaryPen2 * np.sum(self.QuadRules.boundary_w * (wBdry*(self.QuadRules.boundary_t_x*NNbdryphi1dx_idx + self.QuadRules.boundary_t_y*NNbdryphi1dy_idx)) * 
                                                                           (wBdry*(self.QuadRules.boundary_t_x*NNbdryphi1dx_jdx + self.QuadRules.boundary_t_y*NNbdryphi1dy_jdx)), axis=0)

        NNbdryphi2_idx = np.reshape(NN["u2"]["boundary_value"][:,i], [self.QuadRules.boundary_x.shape[0], 1])
        NNbdryphi2_jdx = NN["u2"]["boundary_value"]

        NNbdryphi2dx_idx = np.reshape(NN["u2"]["boundary_x"][:,i], [self.QuadRules.boundary_x.shape[0], 1])
        NNbdryphi2dx_jdx = NN["u2"]["boundary_x"]

        NNbdryphi2dy_idx = np.reshape(NN["u2"]["boundary_y"][:,i], [self.QuadRules.boundary_x.shape[0], 1])
        NNbdryphi2dy_jdx = NN["u2"]["boundary_y"]

        A22[i,:] += self.boundaryPen1 * np.sum(self.QuadRules.boundary_w * (wBdry*NNbdryphi2_idx) * (wBdry*NNbdryphi2_jdx), axis=0)
        A22[i,:] += self.boundaryPen2 * np.sum(self.QuadRules.boundary_w * (wBdry*(self.QuadRules.boundary_t_x*NNbdryphi2dx_idx + self.QuadRules.boundary_t_y*NNbdryphi2dy_idx)) * 
                                                                           (wBdry*(self.QuadRules.boundary_t_x*NNbdryphi2dx_jdx + self.QuadRules.boundary_t_y*NNbdryphi2dy_jdx)), axis=0)

        # RHS
        Fphi1[i] = np.sum(self.QuadRules.interior_w * (w*DATA["F1"]) * (w*(-NNdx2phi1_idx - NNdy2phi1_idx)))
        Fphi1[i] += np.sum(self.QuadRules.interior_w * (w0*DATA["F3"]) * (w0*(NNdxphi1_idx)))
        Fphi1[i] += self.H1flag * np.sum(self.QuadRules.interior_w * (w*DATA["F3x"]) * (w*(NNdx2phi1_idx)))
        Fphi1[i] += self.H1flag * np.sum(self.QuadRules.interior_w * (w*DATA["F3y"]) * (w*(NNdxyphi1_idx)))
        Fphi1[i] += self.boundaryPen1 * np.sum(self.QuadRules.boundary_w * (wBdry*NNbdryphi1_idx) * (wBdry*DATA["G1"]))
        Fphi1[i] += self.boundaryPen2 * np.sum(self.QuadRules.boundary_w * (wBdry*(self.QuadRules.boundary_t_x*NNbdryphi1dx_idx + self.QuadRules.boundary_t_y*NNbdryphi1dy_idx)) * 
                                                                           (wBdry*(self.QuadRules.boundary_t_x*DATA["G1x"] + self.QuadRules.boundary_t_y*DATA["G1y"])))
        Fphi1[i] += -np.sum(self.QuadRules.interior_w * (w*(-NNdx2phi1_idx - NNdy2phi1_idx)) * 
                                                        (w*(-solution0["u1"]["interior_xx"] - solution0["u1"]["interior_yy"] + solution0["p"]["interior_x"])))
        Fphi1[i] += -np.sum(self.QuadRules.interior_w * (w0*NNdxphi1_idx) * (w0*(solution0["u1"]["interior_x"] + solution0["u2"]["interior_y"])))
        Fphi1[i] += -self.H1flag * np.sum(self.QuadRules.interior_w * (w*NNdx2phi1_idx) * (w*(solution0["u1"]["interior_xx"] + solution0["u2"]["interior_yx"])))
        Fphi1[i] += -self.H1flag * np.sum(self.QuadRules.interior_w * (w*NNdxyphi1_idx) * (w*(solution0["u1"]["interior_xy"] + solution0["u2"]["interior_yy"])))
        Fphi1[i] += -self.boundaryPen1 * np.sum(self.QuadRules.boundary_w * (wBdry*NNbdryphi1_idx) * (wBdry*solution0["u1"]["boundary_value"]))
        Fphi1[i] += -self.boundaryPen2 * np.sum(self.QuadRules.boundary_w * (wBdry*(self.QuadRules.boundary_t_x*NNbdryphi1dx_idx + self.QuadRules.boundary_t_y*NNbdryphi1dy_idx)) * 
                                                                            (wBdry*(self.QuadRules.boundary_t_x*solution0["u1"]["boundary_x"] + self.QuadRules.boundary_t_y*solution0["u1"]["boundary_y"])))

        Fphi2[i] = np.sum(self.QuadRules.interior_w * (w*DATA["F2"]) * (w*(-NNdx2phi2_idx - NNdy2phi2_idx)))
        Fphi2[i] += np.sum(self.QuadRules.interior_w * (w0*DATA["F3"]) * (w0*(NNdyphi2_idx)))
        Fphi2[i] += self.H1flag * np.sum(self.QuadRules.interior_w * (w*DATA["F3x"]) * (w*(NNdyxphi2_idx)))
        Fphi2[i] += self.H1flag * np.sum(self.QuadRules.interior_w * (w*DATA["F3y"]) * (w*(NNdy2phi2_idx)))
        Fphi2[i] += self.boundaryPen1 * np.sum(self.QuadRules.boundary_w * (wBdry*NNbdryphi2_idx) * (wBdry*DATA["G2"]))
        Fphi2[i] += self.boundaryPen2 * np.sum(self.QuadRules.boundary_w * (wBdry*(self.QuadRules.boundary_t_x*NNbdryphi2dx_idx + self.QuadRules.boundary_t_y*NNbdryphi2dy_idx)) * 
                                                                           (wBdry*(self.QuadRules.boundary_t_x*DATA["G2x"] + self.QuadRules.boundary_t_y*DATA["G2y"])))
        Fphi2[i] += -np.sum(self.QuadRules.interior_w * (w*(-NNdx2phi2_idx - NNdy2phi2_idx)) * 
                                                        (w*(-solution0["u2"]["interior_xx"] - solution0["u2"]["interior_yy"] + solution0["p"]["interior_y"])))
        Fphi2[i] += -np.sum(self.QuadRules.interior_w * (w0*NNdyphi2_idx) * (w0*(solution0["u1"]["interior_x"] + solution0["u2"]["interior_y"])))
        Fphi2[i] += -self.H1flag * np.sum(self.QuadRules.interior_w * (w*NNdyxphi2_idx) * (w*(solution0["u1"]["interior_xx"] + solution0["u2"]["interior_yx"])))
        Fphi2[i] += -self.H1flag * np.sum(self.QuadRules.interior_w * (w*NNdy2phi2_idx) * (w*(solution0["u1"]["interior_xy"] + solution0["u2"]["interior_yy"])))
        Fphi2[i] += -self.boundaryPen1 * np.sum(self.QuadRules.boundary_w * (wBdry*NNbdryphi2_idx) * (wBdry*solution0["u2"]["boundary_value"]))
        Fphi2[i] += -self.boundaryPen2 * np.sum(self.QuadRules.boundary_w * (wBdry*(self.QuadRules.boundary_t_x*NNbdryphi2dx_idx + self.QuadRules.boundary_t_y*NNbdryphi2dy_idx)) * 
                                                                            (wBdry*(self.QuadRules.boundary_t_x*solution0["u2"]["boundary_x"] + self.QuadRules.boundary_t_y*solution0["u2"]["boundary_y"])))

        return


    def activMatrix2(self, i, k, NN, solution0, C, Fq, DATA):

        xTrain = np.reshape(self.QuadRules.interior_x[:,0], [self.QuadRules.interior_x.shape[0], 1])
        yTrain = np.reshape(self.QuadRules.interior_x[:,1], [self.QuadRules.interior_x.shape[0], 1])

        xCompat = np.reshape(self.QuadRules.compatibility_x[:,0], [self.QuadRules.compatibility_x.shape[0], 1])
        yCompat = np.reshape(self.QuadRules.compatibility_x[:,1], [self.QuadRules.compatibility_x.shape[0], 1])

        if (self.isExtendedNetwork):
            w1 = np.sqrt((xTrain-1.0)**2 + (yTrain)**2)**self.beta
            w2 = np.sqrt((xTrain+1.0)**2 + (yTrain)**2)**self.beta
            w = w1 * w2
        else:
            w = 1.0

        # interior terms
        NNdxq_idx = np.reshape(NN["p"]["interior_x"][:,i], [self.QuadRules.interior_x.shape[0], 1])
        NNdxq_jdx = NN["p"]["interior_x"]

        NNdyq_idx = np.reshape(NN["p"]["interior_y"][:,i], [self.QuadRules.interior_x.shape[0], 1])
        NNdyq_jdx = NN["p"]["interior_y"]

        # C matrix
        C[i,:] = np.sum(self.QuadRules.interior_w * (w*NNdxq_idx) * (w*NNdxq_jdx), axis=0)
        C[i,:] += np.sum(self.QuadRules.interior_w * (w*NNdyq_idx) * (w*NNdyq_jdx), axis=0)

        NNcompat_idx = np.reshape(NN["p"]["compatibility_value"][:,i], [self.QuadRules.compatibility_x.shape[0], 1])
        NNcompat_jdx = NN["p"]["compatibility_value"]

        C[i,:] += self.boundaryPen1 * np.sum(self.QuadRules.compatibility_w * NNcompat_idx * NNcompat_jdx, axis=0)

        # RHS
        Fq[i] = np.sum(self.QuadRules.interior_w * (w*DATA["F1"]) * (w*(NNdxq_idx)))
        Fq[i] += np.sum(self.QuadRules.interior_w * (w*DATA["F2"]) * (w*(NNdyq_idx)))
        Fq[i] += np.sum(self.QuadRules.compatibility_w * NNcompat_idx * DATA["I"])
        Fq[i] += -np.sum(self.QuadRules.interior_w * (w*(NNdxq_idx)) * (w*(-solution0["u1"]["interior_xx"] - solution0["u1"]["interior_yy"] + solution0["p"]["interior_x"])))
        Fq[i] += -np.sum(self.QuadRules.interior_w * (w*(NNdyq_idx)) * (w*(-solution0["u2"]["interior_xx"] - solution0["u2"]["interior_yy"] + solution0["p"]["interior_y"])))
        Fq[i] += -self.boundaryPen1 * np.sum(self.QuadRules.compatibility_w * NNcompat_idx * solution0["p"]["compatibility_value"])

        return


    def galerkinUpdate(self, k, neurons, params, solution0, scale, DATA):
        """
        Routine which computes the subspace Φ = span{Ψ_i}, where Ψ_i (i=1,...,n) are the n components 
        of the last hidden layer of the DNN (plus potential singular functions if using the extended architecture).
        The components Ψ_i and its derivatives in the interior of the domain and on its boundary 
        are computed and stored in the dictionary NNoutput. 

        Returns the linear coefficients of the (linear) activation layer of the DNN by performing
        Galerkin orthogonal projection of the error u-solution0 onto Φ.
        """
        params_i = self.separateParams(params)

        if (self.isExtendedNetwork):
            dim = neurons + 2 * self.alphaSize + 2
        else:
            dim = neurons

        N = self.QuadRules.interior_x.shape[0]
        Nbdry = self.QuadRules.boundary_x.shape[0]
        A11 = np.zeros([dim, dim])
        A12 = np.zeros([dim, dim])
        A22 = np.zeros([dim, dim])
        B1 = np.zeros([dim, dim])
        B2 = np.zeros([dim, dim])
        C = np.zeros([dim, dim])
        Fphi1 = np.zeros([dim, 1])
        Fphi2 = np.zeros([dim, 1])
        Fq = np.zeros([dim, 1])
  
        # compute the derivatives as needed in the interior
        xTrain = np.reshape(self.QuadRules.interior_x[:,0], [self.QuadRules.interior_x.shape[0], 1])
        yTrain = np.reshape(self.QuadRules.interior_x[:,1], [self.QuadRules.interior_x.shape[0], 1])
        xBdryTrain = np.reshape(self.QuadRules.boundary_x[:,0], [self.QuadRules.boundary_x.shape[0], 1])
        yBdryTrain = np.reshape(self.QuadRules.boundary_x[:,1], [self.QuadRules.boundary_x.shape[0], 1])
        xCompat = np.reshape(self.QuadRules.compatibility_x[:,0], [self.QuadRules.compatibility_x.shape[0], 1])
        yCompat = np.reshape(self.QuadRules.compatibility_x[:,1], [self.QuadRules.compatibility_x.shape[0], 1])

        NNoutput = {}
        for field in self.solution_fields:
            NNoutput[field] = {}

            # value of the basis function
            NNoutput[field]["interior_value"] = jnp.squeeze(jax.vmap(self.networkArrayReg, (0, 0, None, None, None),  0)(xTrain, yTrain, params_i[field], scale, field))
            for der in self.intDerivatives[field]:	
                # compute the requested derivative
                der_iter = 0
                for der_i in der:
                    if (der_iter == 0):
                        grad_i = jax.vmap(jax.jacfwd(self.networkArrayReg, int(der_i)), (0, 0, None, None, None), 0)
                    else:
                        grad_i = jax.vmap(jax.jacfwd(grad_i, int(der_i)), (0, 0, None, None, None), 0)
                    der_iter += 1
                der_print = self.parseDerivativeString(der)
                # value of its derivatives
                NNoutput[field]["interior_" + der_print] = jnp.squeeze(grad_i(xTrain, yTrain, params_i[field], scale, field))

       
            # compute corresponding derivatives on the spatial boundary  
            if self.hasSpatialBoundary[field]:
                NNoutput[field]["boundary_value"] = jnp.squeeze(jax.vmap(self.networkArrayReg, (0, 0, None, None, None),  0)(xBdryTrain, yBdryTrain, params_i[field], scale, field))
    
            for der in self.bdryDerivatives[field]:
                # compute the requested derivative
                der_iter = 0
                for der_i in der:
                    if (der_iter == 0):
                        grad_i = jax.vmap(jax.jacfwd(self.networkArrayReg, int(der_i)), (0, 0, None, None, None), 0)
                    else:
                        grad_i = jax.vmap(jax.jacfwd(grad_i, int(der_i)), (0, 0, None, None, None), 0)
                    der_iter += 1
                der_print = self.parseDerivativeString(der)
                NNoutput[field]["boundary_" + der_print] = jnp.squeeze(grad_i(xBdryTrain, yBdryTrain, params_i[field], scale, field))

            # compute compatibility point
            if self.hasCompatibilityCondition[field]:
                NNoutput[field]["compatibility_value"] = jnp.reshape(jax.vmap(self.networkArrayReg, (0, 0, None, None, None), 0)(xCompat, yCompat, params_i[field], scale, field), [1, dim])
    
        # assemble matrices
        t0 = time.time()
        Parallel(n_jobs=8, backend="threading")(
            delayed(self.activMatrix)(i, k, NNoutput, solution0,
                                      A11, A12, A22, B1, B2, Fphi1, Fphi2, DATA) for i in range(dim))

        Parallel(n_jobs=8, backend="threading")(
            delayed(self.activMatrix2)(i, k, NNoutput, solution0, C, Fq, DATA) for i in range(dim))
        t1 = time.time()

        if self.isExtendedNetwork:
            start = 0 
            end = (neurons + 2 * self.alphaSize)

            A1b = A11[start:end, end:(end + 2)] + A12[start:end, end:(end + 2)]
            A2b = A22[start:end, end:(end + 2)] + A12[end:(end + 2), start:end].T
            Abb = A11[end:(end + 2), end:(end + 2)] + A22[end:(end + 2), end:(end + 2)] + A12[end:(end + 2), end:(end + 2)] + A12.T[end:(end + 2), end:(end + 2)]
            Bb = B1[end:(end + 2), :] + B2[end:(end + 2), :]
            K = np.concatenate( (np.concatenate((A11[start:end, start:end], A12[start:end, start:end], A1b, B1[start:end, :]), axis=1), 
                                 np.concatenate((A12[start:end, start:end].T, A22[start:end, start:end], A2b, B2[start:end, :]), axis=1),
                                 np.concatenate((A1b.T, A2b.T, Abb, Bb), axis=1),
                                 np.concatenate((B1.T[:, start:end], B2.T[:, start:end], Bb.T, C), axis=1)), axis=0)
            F = np.concatenate((Fphi1[start:end], Fphi2[start:end], Fphi1[end:(end + 2)] + Fphi2[end:(end + 2)], Fq), axis=0)

            c, _, _, _ = scipy.linalg.lstsq(K, F, cond=None, lapack_driver='gelsy', check_finite=False)
            # c = scipy.linalg.(K, F, lapack_driver='gelsy', check_finite=False)
            c = np.reshape(c, [3*dim - 2,1])
        else:
            K = np.concatenate( (np.concatenate((A11, A12, B1), axis=1), 
                                np.concatenate((A12.T, A22, B2), axis=1),
                                np.concatenate((B1.T, B2.T, C), axis=1)), axis=0)
            F = np.concatenate((Fphi1, Fphi2, Fq), axis=0)
  
            c, _, _, _ = scipy.linalg.lstsq(K, F, cond=None, lapack_driver='gelsy', check_finite=False)
            # c = scipy.linalg.(K, F, lapack_driver='gelsy', check_finite=False)
            c = np.reshape(c, [3*dim,1])


        myCond = np.linalg.cond(K)
        print("Time elapsed for assembly: ", t1-t0)

        if self.isExtendedNetwork:
            dim_coupled = neurons + 2 * self.alphaSize
            cphi1 = np.reshape(c[0:dim_coupled, 0], [dim_coupled, 1])
            cphi2 = np.reshape(c[dim_coupled:2*dim_coupled, 0], [dim_coupled, 1])
            cbottom = np.reshape(c[2*dim_coupled : (2*dim_coupled + 2), 0], [2, 1])
            cq = np.reshape(c[2*dim_coupled + 2 : (3*dim_coupled + 4), 0], [dim_coupled + 2, 1])

            cphi1 = np.concatenate((cphi1, cbottom), axis=0)
            cphi2 = np.concatenate((cphi2, cbottom), axis=0)
        else:
            cphi1 = np.reshape(c[0:dim, 0], [dim, 1])
            cphi2 = np.reshape(c[dim:2*dim, 0], [dim, 1])
            cq = np.reshape(c[2*dim:3*dim, 0], [dim, 1])

        return cphi1, cphi2, cq, myCond


    # @partial(jax.jit, static_argnums=(0,2,))
    def computeLoss(self, params, k, 
                    X, Y, W, 
                    Xbdry, Ybdry, Wbdry, 
                    Xcompat, Ycompat, Wcompat,
                    TX, TY, 
                    activationcoeffs, activationcoeffs_singular, scalek, 
                    SOL0, F1, F2, F3, F3x, F3y,
                    U10bdryexact, U20bdryexact, P0compatexact):
        """
        Evaluates the loss function which is given by

                Loss[v] = -|<r(u0), v>| / a(v,v)^{1/2}.

        The minimum of the loss function (when v is taken over the infinite-dimensional space X) is
        the error u-u0. The minimizer φ of the loss is used as a basis function for a finite-dimensional
        subspace S_i := span{u0, φ1, ..., φn}. The basis functions are corrections to the initial coarse 
        approximation u0.
        """
        params_i = self.separateParams(params)

        # compute required interior fields
        opt_args = {}
        if (self.isExtendedNetwork):
            Calpha = {}
            Calpha["leftcorner"] = {}
            Calpha["rightcorner"] = {}
            Calpha["eddies"] = {}
            for field in self.solution_fields:
                Calpha["leftcorner"][field] = activationcoeffs_singular[field][0 : self.alphaSize]
                Calpha["rightcorner"][field] = activationcoeffs_singular[field][self.alphaSize : 2*self.alphaSize]
                Calpha["eddies"][field] = activationcoeffs_singular[field][2*self.alphaSize : (2*self.alphaSize + 2)]
            opt_args["coeffs"] = Calpha
        else:
            opt_args["coeffs"] = None
             
        phi = {}
        for field in self.solution_fields:
            phi[field] = {}

            phi[field]["interior_value"] = self.createNetwork(X, Y, params_i[field], activationcoeffs[field], scalek, field, opt_args["coeffs"])
            for der in self.intDerivatives[field]:
                der_iter = 0
                for der_i in der:
                    if (der_iter == 0):
                        grad_i = jax.grad(self.createNetwork, int(der_i))
                    else:
                        grad_i = jax.grad(grad_i, int(der_i))
                    der_iter += 1
                der_print = self.parseDerivativeString(der)
                phi[field]["interior_" + der_print] = jax.vmap(grad_i, (0, 0, None, None, None, None, None))(X, Y, params_i[field], activationcoeffs[field], scalek, field, opt_args["coeffs"])

            # compute required boundary fields 
            if self.hasSpatialBoundary[field]:
                phi[field]["boundary_value"] = self.createNetwork(Xbdry, Ybdry, params_i[field], activationcoeffs[field], scalek, field, opt_args["coeffs"])

            for der in self.bdryDerivatives[field]:
                der_iter = 0
                for der_i in der:
                    if (der_iter == 0):
                        grad_i = jax.grad(self.createNetwork, int(der_i))
                    else:
                        grad_i = jax.grad(grad_i, int(der_i))
                    der_iter += 1
                der_print = self.parseDerivativeString(der)
                phi[field]["boundary_" + der_print] = jax.vmap(grad_i, (0, 0, None, None, None, None, None))(Xbdry, Ybdry, params_i[field], activationcoeffs[field], scalek, field, opt_args["coeffs"])

            # compatibility condition
            if self.hasCompatibilityCondition[field]:
                phi[field]["compatibility_value"] = self.createNetwork(Xcompat, Ycompat, params_i[field], activationcoeffs[field], scalek, field, opt_args["coeffs"])
   
        # compute loss function
        if (self.isExtendedNetwork):
            resWeight1 = jnp.power(jnp.sqrt(jnp.power(X-1.0, 2) + jnp.power(Y, 2)), self.beta)
            resWeight2 = jnp.power(jnp.sqrt(jnp.power(X+1.0, 2) + jnp.power(Y, 2)), self.beta)
            resWeight = jnp.multiply(resWeight1, resWeight2)
            resWeight0 = resWeight

            resWeight1 = jnp.power(jnp.sqrt(jnp.power(Xbdry-1.0, 2) + jnp.power(Ybdry, 2)), self.beta)
            resWeight2 = jnp.power(jnp.sqrt(jnp.power(Xbdry+1.0, 2) + jnp.power(Ybdry, 2)), self.beta)
            resWeightBdry = jnp.multiply(resWeight1, resWeight2)
        else:
            resWeight = 1.0
            resWeight0 = 1.0
            resWeightBdry = 1.0

        # compute the energy norm sqrt(a(phi, phi))
        normint1 = jnp.sum(jnp.multiply(W, jnp.square(jnp.multiply(resWeight, -phi["u1"]["interior_xx"]-phi["u1"]["interior_yy"] + phi["p"]["interior_x"]))))
        normint2 = jnp.sum(jnp.multiply(W, jnp.square(jnp.multiply(resWeight, -phi["u2"]["interior_xx"]-phi["u2"]["interior_yy"] + phi["p"]["interior_y"]))))
        normint3 = jnp.sum(jnp.multiply(W, jnp.square(jnp.multiply(resWeight0, phi["u1"]["interior_x"] + phi["u2"]["interior_y"]))))
        normint4 = jnp.multiply(self.H1flag, jnp.sum(jnp.multiply(W, jnp.square(jnp.multiply(resWeight, phi["u1"]["interior_xx"] + phi["u2"]["interior_yx"])))))
        normint5 = jnp.multiply(self.H1flag, jnp.sum(jnp.multiply(W, jnp.square(jnp.multiply(resWeight, phi["u1"]["interior_xy"] + phi["u2"]["interior_yy"])))))
        normbdry1 = jnp.multiply(self.boundaryPen1, jnp.sum(jnp.multiply(Wbdry, jnp.square(resWeightBdry*phi["u1"]["boundary_value"]))))
        normbdry2 = jnp.multiply(self.boundaryPen1, jnp.sum(jnp.multiply(Wbdry, jnp.square(resWeightBdry*phi["u2"]["boundary_value"]))))
        normbdry3 = jnp.multiply(self.boundaryPen2, jnp.sum(jnp.multiply(Wbdry, jnp.square(resWeightBdry*(jnp.multiply(TX, phi["u1"]["boundary_x"]) + jnp.multiply(TY, phi["u1"]["boundary_y"]))))))
        normbdry4 = jnp.multiply(self.boundaryPen2, jnp.sum(jnp.multiply(Wbdry, jnp.square(resWeightBdry*(jnp.multiply(TX, phi["u2"]["boundary_x"]) + jnp.multiply(TY, phi["u2"]["boundary_y"]))))))
        normcompat = jnp.multiply(self.boundaryPen1, jnp.sum(jnp.multiply(Wcompat, jnp.square(phi["p"]["compatibility_value"]))))
        norm = jnp.sqrt(normint1 + normint2 + normint3 + (normint4 + normint5) + normbdry1 + normbdry2 + normbdry3 + normbdry4 + normcompat)
        

        # compute the value of a(u0,phi)
        r1 = jnp.sum(jnp.multiply(W, jnp.multiply(jnp.multiply(resWeight, -SOL0["u1"]["interior_xx"][:,0]-SOL0["u1"]["interior_yy"][:,0] + SOL0["p"]["interior_x"][:,0]), 
                                                  jnp.multiply(resWeight, -phi["u1"]["interior_xx"]-phi["u1"]["interior_yy"] + phi["p"]["interior_x"]))))
        r2 = jnp.sum(jnp.multiply(W, jnp.multiply(jnp.multiply(resWeight, -SOL0["u2"]["interior_xx"][:,0]-SOL0["u2"]["interior_yy"][:,0] + SOL0["p"]["interior_y"][:,0]), 
                                                   jnp.multiply(resWeight, -phi["u2"]["interior_xx"]-phi["u2"]["interior_yy"] + phi["p"]["interior_y"]))))
        r3 = jnp.sum(jnp.multiply(W, jnp.multiply(jnp.multiply(resWeight0, SOL0["u1"]["interior_x"][:,0] + SOL0["u2"]["interior_y"][:,0]), jnp.multiply(resWeight0, phi["u1"]["interior_x"] + phi["u2"]["interior_y"]))))
        r4 = jnp.multiply(self.H1flag, jnp.sum(jnp.multiply(W, jnp.multiply(jnp.multiply(resWeight, SOL0["u1"]["interior_xx"][:,0] + SOL0["u2"]["interior_yx"][:,0]), jnp.multiply(resWeight, phi["u1"]["interior_xx"] + phi["u2"]["interior_yx"])))))
        r5 = jnp.multiply(self.H1flag, jnp.sum(jnp.multiply(W, jnp.multiply(jnp.multiply(resWeight, SOL0["u1"]["interior_xy"][:,0] + SOL0["u2"]["interior_yy"][:,0]), jnp.multiply(resWeight, phi["u1"]["interior_xy"] + phi["u2"]["interior_yy"])))))
        r6 = jnp.multiply(self.boundaryPen1, jnp.sum(jnp.multiply(Wbdry, jnp.multiply(resWeightBdry*SOL0["u1"]["boundary_value"][:,0], resWeightBdry*phi["u1"]["boundary_value"]))))
        r7 = jnp.multiply(self.boundaryPen1, jnp.sum(jnp.multiply(Wbdry, jnp.multiply(resWeightBdry*SOL0["u2"]["boundary_value"][:,0], resWeightBdry*phi["u2"]["boundary_value"]))))
        r8 = jnp.multiply(self.boundaryPen2, jnp.sum(jnp.multiply(Wbdry, jnp.multiply(resWeightBdry*(jnp.multiply(TX, SOL0["u1"]["boundary_x"][:,0]) + jnp.multiply(TY, SOL0["u1"]["boundary_y"][:,0])), 
                                                                                      resWeightBdry*(jnp.multiply(TX, phi["u1"]["boundary_x"]) + jnp.multiply(TY, phi["u1"]["boundary_y"]))))))
        r9 = jnp.multiply(self.boundaryPen2, jnp.sum(jnp.multiply(Wbdry, jnp.multiply(resWeightBdry*(jnp.multiply(TX, SOL0["u2"]["boundary_x"][:,0]) + jnp.multiply(TY, SOL0["u2"]["boundary_y"][:,0])), 
                                                                                      resWeightBdry*(jnp.multiply(TX, phi["u2"]["boundary_x"]) + jnp.multiply(TY, phi["u2"]["boundary_y"]))))))
        r10 = jnp.multiply(self.boundaryPen1, jnp.sum(jnp.multiply(Wcompat, jnp.multiply(SOL0["p"]["compatibility_value"], phi["p"]["compatibility_value"]))))

        # compute the value of L(phi)
        r11 = jnp.sum(jnp.multiply(W, jnp.multiply(jnp.multiply(resWeight, F1), jnp.multiply(resWeight, -phi["u1"]["interior_xx"]-phi["u1"]["interior_yy"] + phi["p"]["interior_x"]))))
        r12 = jnp.sum(jnp.multiply(W, jnp.multiply(jnp.multiply(resWeight, F2), jnp.multiply(resWeight, -phi["u2"]["interior_xx"]-phi["u2"]["interior_yy"] + phi["p"]["interior_y"]))))
        r13 = jnp.sum(jnp.multiply(W, jnp.multiply(jnp.multiply(resWeight0, F3), jnp.multiply(resWeight0, phi["u1"]["interior_x"] + phi["u2"]["interior_y"]))))
        r14 = jnp.multiply(self.H1flag, jnp.sum(jnp.multiply(W, jnp.multiply(jnp.multiply(resWeight, F3x), jnp.multiply(resWeight, phi["u1"]["interior_xx"] + phi["u2"]["interior_yx"])))))
        r15 = jnp.multiply(self.H1flag, jnp.sum(jnp.multiply(W, jnp.multiply(jnp.multiply(resWeight, F3y), jnp.multiply(resWeight, phi["u1"]["interior_xy"] + phi["u2"]["interior_yy"])))))
        r16 = jnp.multiply(self.boundaryPen1, jnp.sum(jnp.multiply(Wbdry, jnp.multiply(resWeightBdry*U10bdryexact["value"], resWeightBdry*phi["u1"]["boundary_value"]))))
        r17 = jnp.multiply(self.boundaryPen1, jnp.sum(jnp.multiply(Wbdry, jnp.multiply(resWeightBdry*U20bdryexact["value"], resWeightBdry*phi["u2"]["boundary_value"]))))
        r18 = jnp.multiply(self.boundaryPen2, jnp.sum(jnp.multiply(Wbdry, jnp.multiply(resWeightBdry*(jnp.multiply(TX, U10bdryexact["x"]) + jnp.multiply(TY, U10bdryexact["y"])), 
                                                                                       resWeightBdry*(jnp.multiply(TX, phi["u1"]["boundary_x"]) + jnp.multiply(TY, phi["u1"]["boundary_y"]))))))
        r19 = jnp.multiply(self.boundaryPen2, jnp.sum(jnp.multiply(Wbdry, jnp.multiply(resWeightBdry*(jnp.multiply(TX, U20bdryexact["x"]) + jnp.multiply(TY, U20bdryexact["y"])), 
                                                                                       resWeightBdry*(jnp.multiply(TX, phi["u2"]["boundary_x"]) + jnp.multiply(TY, phi["u2"]["boundary_y"]))))))
        r20 = jnp.multiply(self.boundaryPen1, jnp.sum(jnp.multiply(Wcompat, jnp.multiply(P0compatexact, phi["p"]["compatibility_value"]))))
        res = r11+r12+r13+r14+r15 + (r16+r17) +r18+r19+r20 - (r1+r2+r3 + (r4+r5) + r6+r7+r8+r9 + r10)

        Loss = jnp.multiply(-1.0, jnp.abs(jnp.divide(res, norm)))

        return jnp.squeeze(Loss)


    def computeError(self, solution0,):
        """
        Computes the errors in the energy norm (always exactly computable even if closed form of solution is unknown)
        and L2 norm (assuming closed form of solution is known).
        """
        # compute Energy and L2 error
        xVal = np.reshape(self.QuadRulesVal.interior_x[:,0], [self.QuadRulesVal.interior_x.shape[0], 1])
        yVal = np.reshape(self.QuadRulesVal.interior_x[:,1], [self.QuadRulesVal.interior_x.shape[0], 1])

        f1, f2, f3, f3x, f3y = self.sourceFunc(xVal, yVal)

        xBdryVal = np.reshape(self.QuadRulesVal.boundary_x[:,0], [self.QuadRulesVal.boundary_x.shape[0], 1])
        yBdryVal = np.reshape(self.QuadRulesVal.boundary_x[:,1], [self.QuadRulesVal.boundary_x.shape[0], 1])

        if (self.isExtendedNetwork):
            resWeight1 = np.sqrt((xVal-1.0)**2 + (yVal)**2)**(self.beta)
            resWeight2 = np.sqrt((xVal+1.0)**2 + (yVal)**2)**(self.beta)
            resWeight = resWeight1 * resWeight2
            resWeight0 = resWeight 

            resWeight1 = np.sqrt((xBdryVal-1.0)**2 + (yBdryVal)**2)**(self.beta)
            resWeight2 = np.sqrt((xBdryVal+1.0)**2 + (yBdryVal)**2)**(self.beta)
            resWeightBdry = resWeight1 * resWeight2
        else:
            resWeight = 1.0
            resWeight0 = 1.0
            resWeightBdry = 1.0

        u1exact, u2exact, pexact = self.exactSol(xVal, yVal)
        L2_k = np.sqrt(np.sum(self.QuadRulesVal.interior_w * ((solution0["u1"]["interior_value"] - u1exact)**2 + (solution0["u2"]["interior_value"] - u2exact)**2)))
        L2_k0 = np.sqrt(np.sum(self.QuadRulesVal.interior_w * (u1exact**2 + u2exact**2)))

        L2_k = L2_k/L2_k0

        L2p_k = np.sqrt(np.sum(self.QuadRulesVal.interior_w * (solution0["p"]["interior_value"] - pexact)**2))
        L2p_k0 = np.sqrt(np.sum(self.QuadRulesVal.interior_w * (pexact)**2))
        L2p_k = L2p_k /L2p_k0

        u1bdryexact, u2bdryexact, _ = self.exactSol(xBdryVal, yBdryVal)
        u1bdryxexact, u1bdryyexact, u2bdryxexact, u2bdryyexact, _, _ = self.exactDx(xBdryVal, yBdryVal)

        xCompatVal = np.reshape(self.QuadRulesVal.compatibility_x[:,0], [self.QuadRulesVal.compatibility_x.shape[0], 1])
        yCompatVal = np.reshape(self.QuadRulesVal.compatibility_x[:,1], [self.QuadRulesVal.compatibility_x.shape[0], 1])
        _, _, pcompatexact = self.exactSol(xCompatVal, yCompatVal)

        exact1 = f1
        exact2 = f2
        exact3 = f3
        exact4 = f3x
        exact5 = f3y
        exact6 = self.QuadRulesVal.boundary_t_x * u1bdryxexact + self.QuadRulesVal.boundary_t_y * u1bdryyexact
        exact7 = self.QuadRulesVal.boundary_t_x * u2bdryxexact + self.QuadRulesVal.boundary_t_y * u2bdryyexact

        approx1 = -solution0["u1"]["interior_xx"] - solution0["u1"]["interior_yy"] + solution0["p"]["interior_x"]
        approx2 = -solution0["u2"]["interior_xx"] - solution0["u2"]["interior_yy"] + solution0["p"]["interior_y"]
        approx3 = solution0["u1"]["interior_x"] + solution0["u2"]["interior_y"]
        approx4 = solution0["u1"]["interior_xx"] + solution0["u2"]["interior_yx"]
        approx5 = solution0["u1"]["interior_xy"] + solution0["u2"]["interior_yy"]
        approx6 = self.QuadRulesVal.boundary_t_x * solution0["u1"]["boundary_x"] + self.QuadRulesVal.boundary_t_y * solution0["u1"]["boundary_y"]
        approx7 = self.QuadRulesVal.boundary_t_x * solution0["u2"]["boundary_x"] + self.QuadRulesVal.boundary_t_y * solution0["u2"]["boundary_y"]

        Energy_k = np.sum(self.QuadRulesVal.interior_w * (resWeight*(exact1 - approx1))**2)
        Energy_k += np.sum(self.QuadRulesVal.interior_w * (resWeight*(exact2 - approx2))**2)
        Energy_k += np.sum(self.QuadRulesVal.interior_w * (resWeight0*(exact3 - approx3))**2)
        Energy_k += self.H1flag * np.sum(self.QuadRulesVal.interior_w * (resWeight*(exact4 - approx4))**2)
        Energy_k += self.H1flag * np.sum(self.QuadRulesVal.interior_w * (resWeight*(exact5 - approx5))**2)
        Energy_k += self.boundaryPen1 * np.sum(self.QuadRulesVal.boundary_w * (resWeightBdry*(solution0["u1"]["boundary_value"] - u1bdryexact))**2)
        Energy_k += self.boundaryPen1 * np.sum(self.QuadRulesVal.boundary_w * (resWeightBdry*(solution0["u2"]["boundary_value"] - u2bdryexact))**2)
        Energy_k += self.boundaryPen2 * np.sum(self.QuadRulesVal.boundary_w * (resWeightBdry*(exact6 - approx6))**2)
        Energy_k += self.boundaryPen2 * np.sum(self.QuadRulesVal.boundary_w * (resWeightBdry*(exact7 - approx7))**2)
        Energy_k += self.boundaryPen1 * np.sum(self.QuadRulesVal.compatibility_w * (pcompatexact - solution0["p"]["compatibility_value"])**2)
        Energy_k = np.sqrt(Energy_k)

        Energy0 = np.sum(self.QuadRulesVal.interior_w * (resWeight*(exact1))**2)
        Energy0 += np.sum(self.QuadRulesVal.interior_w * (resWeight*(exact2))**2)
        Energy0 += np.sum(self.QuadRulesVal.interior_w * (resWeight0*exact3)**2)
        Energy0 += self.H1flag * np.sum(self.QuadRulesVal.interior_w * (resWeight*(exact4))**2)
        Energy0 += self.H1flag * np.sum(self.QuadRulesVal.interior_w * (resWeight*(exact5))**2)
        Energy0 += self.boundaryPen1 * np.sum(self.QuadRulesVal.boundary_w * (resWeightBdry*u1bdryexact)**2)
        Energy0 += self.boundaryPen1 * np.sum(self.QuadRulesVal.boundary_w * (resWeightBdry*u2bdryexact)**2)
        Energy0 += self.boundaryPen2 * np.sum(self.QuadRulesVal.boundary_w * (resWeightBdry*exact6)**2)
        Energy0 += self.boundaryPen2 * np.sum(self.QuadRulesVal.boundary_w * (resWeightBdry*exact7)**2)
        Energy0 += self.boundaryPen1 * np.sum(self.QuadRulesVal.compatibility_w * (pcompatexact)**2)
        Energy0 = np.sqrt(Energy0)
        Energy_k = Energy_k/Energy0

        return L2_k, Energy_k, L2_k0, Energy0, L2p_k


    def appendBasis(self, k, params, activationcoeffs, activationcoeffs_singular, scale):
        """
        Adds the basis function described by params, activationcoeffs, and activationcoeffs_singular to the
        subspace. The basis function and its necessary derivatives are evaluate at the quadrature nodes and
        plotting points.
        """
        params_i = self.separateParams(params)

        xTrain = np.reshape(self.QuadRules.interior_x[:,0], [self.QuadRules.interior_x.shape[0], 1])
        yTrain = np.reshape(self.QuadRules.interior_x[:,1], [self.QuadRules.interior_x.shape[0], 1])
        xBdryTrain = np.reshape(self.QuadRules.boundary_x[:,0], [self.QuadRules.boundary_x.shape[0], 1])
        yBdryTrain = np.reshape(self.QuadRules.boundary_x[:,1], [self.QuadRules.boundary_x.shape[0], 1])
        xCompatTrain = np.reshape(self.QuadRules.compatibility_x[:,0], [self.QuadRules.compatibility_x.shape[0], 1])
        yCompatTrain = np.reshape(self.QuadRules.compatibility_x[:,1], [self.QuadRules.compatibility_x.shape[0], 1])

        xVal = np.reshape(self.QuadRulesVal.interior_x[:,0], [self.QuadRulesVal.interior_x.shape[0], 1])
        yVal = np.reshape(self.QuadRulesVal.interior_x[:,1], [self.QuadRulesVal.interior_x.shape[0], 1])
        xBdryVal = np.reshape(self.QuadRulesVal.boundary_x[:,0], [self.QuadRulesVal.boundary_x.shape[0], 1])
        yBdryVal = np.reshape(self.QuadRulesVal.boundary_x[:,1], [self.QuadRulesVal.boundary_x.shape[0], 1])
        xCompatVal = np.reshape(self.QuadRulesVal.compatibility_x[:,0], [self.QuadRulesVal.compatibility_x.shape[0], 1])
        yCompatVal = np.reshape(self.QuadRulesVal.compatibility_x[:,1], [self.QuadRulesVal.compatibility_x.shape[0], 1])

        xStream = np.reshape(self.xStream0[:,0], [self.xStream0.shape[0], 1])
        yStream = np.reshape(self.xStream0[:,1], [self.xStream0.shape[0], 1])

        opt_args = {}
        if (self.isExtendedNetwork):
            Calpha = {}
            Calpha["leftcorner"] = {}
            Calpha["rightcorner"] = {}
            Calpha["eddies"] = {}
            for field in self.solution_fields:
                Calpha["leftcorner"][field] = activationcoeffs_singular[field][0 : self.alphaSize]
                Calpha["rightcorner"][field] = activationcoeffs_singular[field][self.alphaSize : 2*self.alphaSize]
                Calpha["eddies"][field] = activationcoeffs_singular[field][2*self.alphaSize : (2*self.alphaSize + 2)]
            opt_args["coeffs"] = Calpha
        else:
            opt_args["coeffs"] = None

        # basis evaluated at training points
        for field in self.solution_fields:
            field_basis = {}

            # interior value and derivatives
            field_basis["interior_value"] = self.createNetwork(xTrain, yTrain, params_i[field], activationcoeffs[field], scale, field, opt_args["coeffs"])
            for der in self.intDerivatives[field]:			
                # compute the requested derivative
                der_iter = 0
                for der_i in der:
                    if (der_iter == 0):
                        grad_i = jax.grad(self.createNetwork, int(der_i))
                    else:
                        grad_i = jax.grad(grad_i, int(der_i))
                    der_iter += 1
                der_print = self.parseDerivativeString(der)
                field_basis["interior_" + der_print] = jax.vmap(grad_i, (0, 0, None, None, None, None, None))(xTrain[:,0], yTrain[:,0], params_i[field], activationcoeffs[field], scale, field, opt_args["coeffs"])

            # spatial boundary value and derivatives
            if self.hasSpatialBoundary[field]:
                field_basis["boundary_value"] = self.createNetwork(xBdryTrain, yBdryTrain, params_i[field], activationcoeffs[field], scale, field, opt_args["coeffs"])
            for der in self.bdryDerivatives[field]:			
                # compute the requested derivative
                der_iter = 0
                for der_i in der:
                    if (der_iter == 0):
                        grad_i = jax.grad(self.createNetwork, int(der_i))
                    else:
                        grad_i = jax.grad(grad_i, int(der_i))
                    der_iter += 1
                der_print = self.parseDerivativeString(der)
                field_basis["boundary_" + der_print] = jax.vmap(grad_i, (0, 0, None, None, None, None, None))(xBdryTrain[:,0], yBdryTrain[:,0], params_i[field], activationcoeffs[field], scale, field, opt_args["coeffs"])

            # compatibility condition
            if self.hasCompatibilityCondition[field]:
                field_basis["compatibility_value"] = self.createNetwork(xCompatTrain, yCompatTrain, params_i[field], activationcoeffs[field], scale, field, opt_args["coeffs"])
        
            self.basis[field].append(field_basis)


        # basis evaluated at validation points
        for field in self.solution_fields:
            field_basis = {}

            # interior value and derivatives
            field_basis["interior_value"] = self.createNetwork(xVal, yVal, params_i[field], activationcoeffs[field], scale, field, opt_args["coeffs"])
            for der in self.intDerivatives[field]:			
                # compute the requested derivative
                der_iter = 0
                for der_i in der:
                    if (der_iter == 0):
                        grad_i = jax.grad(self.createNetwork, int(der_i))
                    else:
                        grad_i = jax.grad(grad_i, int(der_i))
                    der_iter += 1
                der_print = self.parseDerivativeString(der)
                field_basis["interior_" + der_print] = jax.vmap(grad_i, (0, 0, None, None, None, None, None))(xVal[:,0], yVal[:,0], params_i[field], activationcoeffs[field], scale, field, opt_args["coeffs"])

            # spatial boundary value and derivatives
            if self.hasSpatialBoundary[field]:
                field_basis["boundary_value"] = self.createNetwork(xBdryVal, yBdryVal, params_i[field], activationcoeffs[field], scale, field, opt_args["coeffs"])
            for der in self.bdryDerivatives[field]:			
                # compute the requested derivative
                der_iter = 0
                for der_i in der:
                    if (der_iter == 0):
                        grad_i = jax.grad(self.createNetwork, int(der_i))
                    else:
                        grad_i = jax.grad(grad_i, int(der_i))
                    der_iter += 1
                der_print = self.parseDerivativeString(der)
                field_basis["boundary_" + der_print] = jax.vmap(grad_i, (0, 0, None, None, None, None, None))(xBdryVal[:,0], yBdryVal[:,0], params_i[field], activationcoeffs[field], scale, field, opt_args["coeffs"])

            # compatibility condition
            if self.hasCompatibilityCondition[field]:
                field_basis["compatibility_value"] = self.createNetwork(xCompatVal, yCompatVal, params_i[field], activationcoeffs[field], scale, field, opt_args["coeffs"])
        
            self.basis_val[field].append(field_basis)


        # and evaluate at the equally-spaced plotting points
        for field in self.solution_fields:
            field_basis = {}
            field_basis["interior_value"] = self.createNetwork(xStream, yStream, params_i[field], activationcoeffs[field], scale, field, opt_args["coeffs"])
            self.basis_stream[field].append(field_basis)

        return

    def galerkinSolve(self, k):
        """
        Solves the variational problem

                u_i S_i : a(u_i,v) = L(v)   v S_i

        using the subspace S_i = span{u0, φ1, ..., φn}. The solution u_i and its derivatives
        are evaluated in the interior and boundary quadrature nodes. 
        """
        # generate best approximation as linear combination of basis functions
        xTrain = np.reshape(self.QuadRules.interior_x[:,0], [self.QuadRules.interior_x.shape[0], 1])
        yTrain = np.reshape(self.QuadRules.interior_x[:,1], [self.QuadRules.interior_x.shape[0], 1])
        xBdryTrain = np.reshape(self.QuadRules.boundary_x[:,0], [self.QuadRules.boundary_x.shape[0], 1])
        yBdryTrain = np.reshape(self.QuadRules.boundary_x[:,1], [self.QuadRules.boundary_x.shape[0], 1])
        xCompatTrain = np.reshape(self.QuadRules.compatibility_x[:,0], [self.QuadRules.compatibility_x.shape[0], 1])
        yCompatTrain = np.reshape(self.QuadRules.compatibility_x[:,1], [self.QuadRules.compatibility_x.shape[0], 1])

        xVal = np.reshape(self.QuadRulesVal.interior_x[:,0], [self.QuadRulesVal.interior_x.shape[0], 1])
        yVal = np.reshape(self.QuadRulesVal.interior_x[:,1], [self.QuadRulesVal.interior_x.shape[0], 1])
        xBdryVal = np.reshape(self.QuadRulesVal.boundary_x[:,0], [self.QuadRulesVal.boundary_x.shape[0], 1])
        yBdryVal = np.reshape(self.QuadRulesVal.boundary_x[:,1], [self.QuadRulesVal.boundary_x.shape[0], 1])
        xCompatVal = np.reshape(self.QuadRulesVal.compatibility_x[:,0], [self.QuadRulesVal.compatibility_x.shape[0], 1])
        yCompatVal = np.reshape(self.QuadRulesVal.compatibility_x[:,1], [self.QuadRulesVal.compatibility_x.shape[0], 1])

        xStream = np.reshape(self.xStream0[:,0], [self.xStream0.shape[0], 1])
        yStream = np.reshape(self.xStream0[:,1], [self.xStream0.shape[0], 1])

        f1Train, f2Train, f3Train, f3xTrain, f3yTrain = self.sourceFunc(xTrain, yTrain)
        G1, G2, _ = self.exactSol(xBdryTrain, yBdryTrain)
        G1x, G1y, G2x, G2y, _, _ = self.exactDx(xBdryTrain, yBdryTrain)
        _, _, I = self.exactSol(xCompatTrain, yCompatTrain)

        f1Train = np.squeeze(f1Train)
        f2Train = np.squeeze(f2Train)
        f3Train = np.squeeze(f3Train)
        f3xTrain = np.squeeze(f3xTrain)
        f3yTrain = np.squeeze(f3yTrain)
        G1 = np.squeeze(G1)
        G2 = np.squeeze(G2)
        G1x = np.squeeze(G1x)
        G1y = np.squeeze(G1y)
        G2x = np.squeeze(G2x)
        G2y = np.squeeze(G2y)
        I = np.squeeze(I)

        A11 = np.zeros([k+1, k+1])
        A12 = np.zeros([k+1, k+1])
        A22 = np.zeros([k+1, k+1])
        B1 = np.zeros([k+1, k+1])
        B2 = np.zeros([k+1, k+1])
        C = np.zeros([k+1, k+1])
        Fu1 = np.zeros([k+1, 1])
        Fu2 = np.zeros([k+1, 1])
        Fp = np.zeros([k+1, 1])

        if (self.isExtendedNetwork):
            w1 = np.squeeze(np.sqrt((xTrain-1.0)**2 + (yTrain)**2)**self.beta)
            w2 = np.squeeze(np.sqrt((xTrain+1.0)**2 + (yTrain)**2)**self.beta)
            w = w1 * w2
            w0 = w 

            w1 = np.squeeze(np.sqrt((xBdryTrain-1.0)**2 + (yBdryTrain)**2)**self.beta)
            w2 = np.squeeze(np.sqrt((xBdryTrain+1.0)**2 + (yBdryTrain)**2)**self.beta)
            wBdry = w1 * w2
        else:
            w = 1.0
            w0 = 1.0
            wBdry = 1.0

        # the sums here are not vectorized so all arrays need to be one-dimensional. the basis functions are already
        # one-dimensional arrays. quadrature rules and exact functions for the RHS need to be converted to 1d
        wGlobalFlattened = np.squeeze(self.QuadRules.interior_w)
        wBdryFlattened = np.squeeze(self.QuadRules.boundary_w)
        wCompatFlattened = np.squeeze(self.QuadRules.compatibility_w)
        txFlattened = np.squeeze(self.QuadRules.boundary_t_x)
        tyFlattened = np.squeeze(self.QuadRules.boundary_t_y)

        # assemble linear system
        for idx in range(k+1):
            for jdx in range(k+1):
                A11[idx,jdx] = np.sum(wGlobalFlattened * (w*(-self.basis["u1"][idx]["interior_xx"] - self.basis["u1"][idx]["interior_yy"])) * 
                                                         (w*(-self.basis["u1"][jdx]["interior_xx"] - self.basis["u1"][jdx]["interior_yy"])))
                A11[idx,jdx] += np.sum(wGlobalFlattened * (w0*self.basis["u1"][idx]["interior_x"]) * (w0*self.basis["u1"][jdx]["interior_x"]))
                A11[idx,jdx] += self.H1flag * np.sum(wGlobalFlattened  * (w*self.basis["u1"][idx]["interior_xx"]) * (w*self.basis["u1"][jdx]["interior_xx"]))
                A11[idx,jdx] += self.H1flag * np.sum(wGlobalFlattened  * (w*self.basis["u1"][idx]["interior_xy"]) * (w*self.basis["u1"][jdx]["interior_xy"]))
                A11[idx,jdx] += self.boundaryPen1 * np.sum(wBdryFlattened * (wBdry*self.basis["u1"][idx]["boundary_value"]) * (wBdry*self.basis["u1"][jdx]["boundary_value"]))
                A11[idx,jdx] += self.boundaryPen2 * np.sum(wBdryFlattened* (wBdry*(txFlattened*self.basis["u1"][idx]["boundary_x"] + tyFlattened*self.basis["u1"][idx]["boundary_y"])) * 
                                                                           (wBdry*(txFlattened*self.basis["u1"][jdx]["boundary_x"] + tyFlattened*self.basis["u1"][jdx]["boundary_y"])))

                A12[idx,jdx] = np.sum(wGlobalFlattened  * (w0*self.basis["u1"][idx]["interior_x"]) * (w0*self.basis["u2"][jdx]["interior_y"]))
                A12[idx,jdx] += self.H1flag * np.sum(wGlobalFlattened  * (w*self.basis["u1"][idx]["interior_xx"]) * (w*self.basis["u2"][jdx]["interior_yx"]))
                A12[idx,jdx] += self.H1flag * np.sum(wGlobalFlattened  * (w*self.basis["u1"][idx]["interior_xy"]) * (w*self.basis["u2"][jdx]["interior_yy"]))

                A22[idx,jdx] = np.sum(wGlobalFlattened  * (w*(-self.basis["u2"][idx]["interior_xx"] - self.basis["u2"][idx]["interior_yy"])) * 
                                                          (w*(-self.basis["u2"][jdx]["interior_xx"] - self.basis["u2"][jdx]["interior_yy"])))
                A22[idx,jdx] += np.sum(wGlobalFlattened  * (w0*self.basis["u2"][idx]["interior_y"]) * (w0*self.basis["u2"][jdx]["interior_y"]))
                A22[idx,jdx] += self.H1flag * np.sum(wGlobalFlattened  * (w*self.basis["u2"][idx]["interior_yx"]) * (w*self.basis["u2"][jdx]["interior_yx"]))
                A22[idx,jdx] += self.H1flag * np.sum(wGlobalFlattened  * (w*self.basis["u2"][idx]["interior_yy"]) * (w*self.basis["u2"][jdx]["interior_yy"]))
                A22[idx,jdx] += self.boundaryPen1 * np.sum(wBdryFlattened * (wBdry*self.basis["u2"][idx]["boundary_value"]) * (wBdry*self.basis["u2"][jdx]["boundary_value"]))
                A22[idx,jdx] += self.boundaryPen2 * np.sum(wBdryFlattened * (wBdry*(txFlattened*self.basis["u2"][idx]["boundary_x"] + tyFlattened*self.basis["u2"][idx]["boundary_y"])) * 
                                                                            (wBdry*(txFlattened*self.basis["u2"][jdx]["boundary_x"] + tyFlattened*self.basis["u2"][jdx]["boundary_y"])))

                B1[idx,jdx] = np.sum(wGlobalFlattened  * (w*(-self.basis["u1"][idx]["interior_xx"] - self.basis["u1"][idx]["interior_yy"])) * 
                                                         (w*self.basis["p"][jdx]["interior_x"]))
                B2[idx,jdx] = np.sum(wGlobalFlattened  * (w*(-self.basis["u2"][idx]["interior_xx"] - self.basis["u2"][idx]["interior_yy"])) * 
                                                         (w*self.basis["p"][jdx]["interior_y"]))

                C[idx,jdx] = np.sum(wGlobalFlattened  * (w*self.basis["p"][idx]["interior_x"]) * (w*self.basis["p"][jdx]["interior_x"]))
                C[idx,jdx] += np.sum(wGlobalFlattened  * (w*self.basis["p"][idx]["interior_y"]) * (w*self.basis["p"][jdx]["interior_y"]))
                C[idx, jdx] += self.boundaryPen1 * np.sum(wCompatFlattened * (self.basis["p"][idx]["compatibility_value"]) * (self.basis["p"][jdx]["compatibility_value"]))

            Fu1[idx] = np.sum(wGlobalFlattened * (w*(-self.basis["u1"][idx]["interior_xx"] - self.basis["u1"][idx]["interior_yy"])) * 
                                                 (w*f1Train))
            Fu1[idx] += np.sum(wGlobalFlattened * (w0*(self.basis["u1"][idx]["interior_x"])) *
                                                  (w0*f3Train))
            Fu1[idx] += self.H1flag * np.sum(wGlobalFlattened * (w*(self.basis["u1"][idx]["interior_xx"])) *
                                                  (w*f3xTrain))
            Fu1[idx] += self.H1flag * np.sum(wGlobalFlattened * (w*(self.basis["u1"][idx]["interior_xy"])) *
                                                  (w*f3yTrain))
            Fu1[idx] += self.boundaryPen1 * np.sum(wBdryFlattened * (wBdry*self.basis["u1"][idx]["boundary_value"]) * (wBdry*G1))
            Fu1[idx] += self.boundaryPen2 * np.sum(wBdryFlattened * (wBdry*(txFlattened*self.basis["u1"][idx]["boundary_x"] + tyFlattened*self.basis["u1"][idx]["boundary_y"])) * 
                                                                    (wBdry*(txFlattened*G1x + tyFlattened*G1y)))

            Fu2[idx] = np.sum(wGlobalFlattened * (w*(-self.basis["u2"][idx]["interior_xx"] - self.basis["u2"][idx]["interior_yy"])) * 
                                                 (w*f2Train))
            Fu2[idx] += np.sum(wGlobalFlattened * (w0*(self.basis["u2"][idx]["interior_y"])) *
                                                  (w0*f3Train))
            Fu2[idx] += self.H1flag * np.sum(wGlobalFlattened * (w*(self.basis["u2"][idx]["interior_yx"])) *
                                                  (w*f3xTrain))
            Fu2[idx] += self.H1flag * np.sum(wGlobalFlattened * (w*(self.basis["u2"][idx]["interior_yy"])) *
                                                  (w*f3yTrain))
            Fu2[idx] += self.boundaryPen1 * np.sum(wBdryFlattened * (wBdry*self.basis["u2"][idx]["boundary_value"]) * (wBdry*G2))
            Fu2[idx] += self.boundaryPen2 * np.sum(wBdryFlattened* (wBdry*(txFlattened*self.basis["u2"][idx]["boundary_x"] + tyFlattened*self.basis["u2"][idx]["boundary_y"])) * 
                                                                   (wBdry*(txFlattened*G2x + tyFlattened*G2y)))

            Fp[idx] = np.sum(wGlobalFlattened * (w*(self.basis["p"][idx]["interior_x"])) * (w*f1Train))
            Fp[idx] += np.sum(wGlobalFlattened * (w*(self.basis["p"][idx]["interior_y"])) * (w*f2Train))
            Fp[idx] += self.boundaryPen1 * np.sum(wCompatFlattened * (self.basis["p"][idx]["compatibility_value"]) * (I))

        A = np.concatenate( (np.concatenate((A11, A12, B1), axis=1), np.concatenate((A12.T, A22, B2), axis=1), np.concatenate((B1.T, B2.T, C), axis=1)), axis=0)
        b = np.concatenate( (Fu1, Fu2, Fp), axis=0)

        # if MATLAB engine is installed, use it
        # c = eng.lsqminnorm(matlab.double(A.tolist()), matlab.double(b.tolist()))
        [c, _, _, _] = scipy.linalg.lstsq(A, b)
        c = np.asarray(c)
        c = np.reshape(c, [3*(k+1),1])

        cu1 = np.reshape(c[0:k+1], [k+1,1])
        cu2 = np.reshape(c[k+1:2*(k+1)], [k+1,1])
        cp = np.reshape(c[2*(k+1):3*(k+1)], [k+1,1])

        c = {}
        c["u1"] = cu1
        c["u2"] = cu2
        c["p"] = cp

        print(cu1, cu2, cp)

        # compute linear combination of basis functions. these need to be 2d arrays with shape [k,1]
        # since activMatrix expects them to be 2d arrays for vectorized computations
        uTrain = {}

        for field in self.solution_fields:
            uTrain[field] = {}

            # interior values and derivatives
            val = np.zeros([xTrain.shape[0], ])
            for idx in range(k+1):
                val += c[field][idx] * self.basis[field][idx]["interior_value"]
            uTrain[field]["interior_value"] = np.reshape(val, xTrain.shape)

            for der in self.intDerivatives[field]:
                der_print = self.parseDerivativeString(der)

                # the sum needs to be a one-dimensional array because self.basis["u1"][idx] are
                # all one-dimensional arrays. convert to 2d array afterwards
                val = np.zeros([xTrain.shape[0], ])
                for idx in range(k+1):
                    val += c[field][idx] * self.basis[field][idx]["interior_" + der_print]
                uTrain[field]["interior_" + der_print] = np.reshape(val, xTrain.shape)

            # spatial boundary values and derivatives
            if self.hasSpatialBoundary[field]:
                val = np.zeros([xBdryTrain.shape[0], ])
                for idx in range(k+1):
                    val += c[field][idx] * self.basis[field][idx]["boundary_value"]
                uTrain[field]["boundary_value"] = np.reshape(val, xBdryTrain.shape)

            for der in self.bdryDerivatives[field]:
                der_print = self.parseDerivativeString(der)
                val = np.zeros([xBdryTrain.shape[0], ])
                for idx in range(k+1):
                    val += c[field][idx] * self.basis[field][idx]["boundary_" + der_print]
                uTrain[field]["boundary_" + der_print] = np.reshape(val, xBdryTrain.shape)

            # compatibility condition
            if self.hasCompatibilityCondition[field]:
                val = np.zeros([xCompatTrain.shape[0], ])
                for idx in range(k+1):
                    val += c[field][idx] * self.basis[field][idx]["compatibility_value"]
                uTrain[field]["compatibility_value"] = np.reshape(val, xCompatTrain.shape)
      
        # repeat for validation points
        uVal = {}

        for field in self.solution_fields:
            uVal[field] = {}

            # interior values and derivatives
            val = np.zeros([xVal.shape[0], ])
            for idx in range(k+1):
                val += c[field][idx] * self.basis_val[field][idx]["interior_value"]
            uVal[field]["interior_value"] = np.reshape(val, xVal.shape)

            for der in self.intDerivatives[field]:
                der_print = self.parseDerivativeString(der)

                # the sum needs to be a one-dimensional array because self.basis["u1"][idx] are
                # all one-dimensional arrays. convert to 2d array afterwards
                val = np.zeros([xVal.shape[0], ])
                for idx in range(k+1):
                    val += c[field][idx] * self.basis_val[field][idx]["interior_" + der_print]
                uVal[field]["interior_" + der_print] = np.reshape(val, xVal.shape)

            # spatial boundary values and derivatives
            if self.hasSpatialBoundary[field]:
                val = np.zeros([xBdryVal.shape[0], ])
                for idx in range(k+1):
                    val += c[field][idx] * self.basis_val[field][idx]["boundary_value"]
                uVal[field]["boundary_value"] = np.reshape(val, xBdryVal.shape)

            for der in self.bdryDerivatives[field]:
                der_print = self.parseDerivativeString(der)
                val = np.zeros([xBdryVal.shape[0], ])
                for idx in range(k+1):
                    val += c[field][idx] * self.basis_val[field][idx]["boundary_" + der_print]
                uVal[field]["boundary_" + der_print] = np.reshape(val, xBdryVal.shape)

            # compatibility condition
            if self.hasCompatibilityCondition[field]:
                val = np.zeros([xCompatVal.shape[0], ])
                for idx in range(k+1):
                    val += c[field][idx] * self.basis_val[field][idx]["compatibility_value"]
                uVal[field]["compatibility_value"] = np.reshape(val, xCompatVal.shape)


        # repeat for equally-spaced plotting points, but only store the solution values
        uStream = {}
  
        for field in self.solution_fields:
            uStream[field] = {}
            val = np.zeros([xStream.shape[0], ])
            for idx in range(k+1):
                val += c[field][idx] * self.basis_stream[field][idx]["interior_value"]
            uStream[field]["interior_value"] = np.reshape(val, xStream.shape)
   
        return (uTrain, uVal, uStream, cu1, cu2, cp)


    def plotSurface(self, k, x, y, z, myTitle, myPath):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap=cm.coolwarm)
        ax.set_box_aspect((3,2,1))

        ax.set_xlabel('$x$', fontsize=16)
        ax.set_ylabel('$y$', fontsize=16)
        ax.set_zlabel('$z$', fontsize=16)
        plt.title(myTitle, fontsize=16)
        plt.savefig(myPath)
        plt.close()

        return


    def plotVelocity(self, k, x, y, z, myTitle, myPath):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        surf = ax.contourf(x, y, np.log10(z), levels=np.arange(-12, 2, 1), cmap=cm.coolwarm)
        # ax.set_box_aspect((2,3,1))

        ax.set_xlabel('$x$', labelpad=10)
        ax.set_ylabel('$y$', labelpad=10)	

        num_ticks = 5
        ax.set_xticks(np.linspace(-2,2,num_ticks))
        ax.set_yticks(np.linspace(-3,2,num_ticks))

        plt.title(myTitle, fontsize=16)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(myPath)
        plt.close()

        return
    

    def update(self, params, k, activationcoeffs, activationcoeffs_singular, scalek, 
               SOL0, F1, F2, F3, F3x, F3y,
               U10bdryexact, U20bdryexact, P0compatexact, 
               opt_state, opt_update, get_params):
        value, grads = jax.value_and_grad(self.computeLoss, argnums=0)(params, k, self.QuadRules.interior_x[:,0], self.QuadRules.interior_x[:,1], self.QuadRules.interior_w[:,0], 
                                                                    self.QuadRules.boundary_x[:,0], self.QuadRules.boundary_x[:,1], self.QuadRules.boundary_w[:,0], 
                                                                    self.QuadRules.compatibility_x[:,0], self.QuadRules.compatibility_x[:,1], self.QuadRules.compatibility_w[:,0], 
                                                                    self.QuadRules.boundary_t_x[:,0], self.QuadRules.boundary_t_y[:,0],  
                                                                    activationcoeffs, activationcoeffs_singular, scalek, 
                                                                    SOL0, F1, F2, F3, F3x, F3y, U10bdryexact, U20bdryexact, P0compatexact)
        opt_state = opt_update(0, grads, opt_state)

        return get_params(opt_state), opt_state, value


    def generateBasis(self):
        """
        Generates a finite-dimensional subspace of dimension maxRefs and solves the variational problem.
        """
        xTrain = np.reshape(self.QuadRules.interior_x[:,0], [self.QuadRules.interior_x.shape[0], 1])
        yTrain = np.reshape(self.QuadRules.interior_x[:,1], [self.QuadRules.interior_x.shape[0], 1])
        xVal = np.reshape(self.QuadRulesVal.interior_x[:,0], [self.QuadRulesVal.interior_x.shape[0], 1])
        yVal = np.reshape(self.QuadRulesVal.interior_x[:,1], [self.QuadRulesVal.interior_x.shape[0], 1])

        xBdryTrain = np.reshape(self.QuadRules.boundary_x[:,0], [self.QuadRules.boundary_x.shape[0], 1])
        yBdryTrain = np.reshape(self.QuadRules.boundary_x[:,1], [self.QuadRules.boundary_x.shape[0], 1])
        xBdryVal = np.reshape(self.QuadRulesVal.boundary_x[:,0], [self.QuadRulesVal.boundary_x.shape[0], 1])
        yBdryVal = np.reshape(self.QuadRulesVal.boundary_x[:,1], [self.QuadRulesVal.boundary_x.shape[0], 1])

        xCompatTrain = np.reshape(self.QuadRules.compatibility_x[:,0], [self.QuadRules.compatibility_x.shape[0], 1])
        yCompatTrain = np.reshape(self.QuadRules.compatibility_x[:,1], [self.QuadRules.compatibility_x.shape[0], 1])
        xCompatVal = np.reshape(self.QuadRulesVal.compatibility_x[:,0], [self.QuadRulesVal.compatibility_x.shape[0], 1])
        yCompatVal = np.reshape(self.QuadRulesVal.compatibility_x[:,1], [self.QuadRulesVal.compatibility_x.shape[0], 1])

        f1Train, f2Train, f3Train, f3xTrain, f3yTrain = self.sourceFunc(xTrain, yTrain)
        f1Val, f2Val, f3Val, f3xVal, f3yVal = self.sourceFunc(xVal, yVal)

        xStream = np.reshape(self.xStream0[:,0], [self.xStream0.shape[0], 1])
        yStream = np.reshape(self.xStream0[:,1], [self.xStream0.shape[0], 1])

        # initial approximation at training points
        uTrain = {}

        for field in self.solution_fields:
            uTrain[field] = {}

            # interior
            uTrain[field]["interior_value"] = np.zeros(xTrain.shape)
            for der in self.intDerivatives[field]:
                der_print = self.parseDerivativeString(der)
                uTrain[field]["interior_" + der_print] = np.zeros(xTrain.shape)

            # spatial boundary
            if self.hasSpatialBoundary[field]:
                uTrain[field]["boundary_value"] = np.zeros(xBdryTrain.shape)
            for der in self.bdryDerivatives[field]:
                der_print = self.parseDerivativeString(der)
                uTrain[field]["boundary_" + der_print] = np.zeros(xBdryTrain.shape)

            # compatibility condition
            if self.hasCompatibilityCondition[field]:
                uTrain[field]["compatibility_value"] = np.zeros(xCompatTrain.shape)
  
        # repeat for validation points
        uVal = {}

        for field in self.solution_fields:
            uVal[field] = {}

            # interiors
            uVal[field]["interior_value"] = np.zeros(xVal.shape)
            for der in self.intDerivatives[field]:
                der_print = self.parseDerivativeString(der)
                uVal[field]["interior_" + der_print] = np.zeros(xVal.shape)
   
            # spatial boundary
            if self.hasSpatialBoundary[field]:
                uVal[field]["boundary_value"] = np.zeros(xBdryVal.shape)
            for der in self.bdryDerivatives[field]:
                der_print = self.parseDerivativeString(der)
                uVal[field]["boundary_" + der_print] = np.zeros(xBdryVal.shape)

            # compatibility
            if self.hasCompatibilityCondition[field]:
                uVal[field]["compatibility_value"] = np.zeros(xCompatVal.shape)

        # repeating for plotting points
        uStream = {}

        for field in self.solution_fields:
            uStream[field] = {}
            uStream[field]["interior_value"] = np.zeros([self.xStream0.shape[0],1])


        # generate new basis
        errorIndicator = 1.e3
        k = 0

        karray = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        neuronsarray = [int(self.neurons * (1.9**ki)) for ki in karray]
        neuronsarray = np.reshape(neuronsarray, [9, 1])

        scalearray = [np.minimum(self.scale + 0.25*ki, 950.0) for ki in karray]
        scalearray = np.reshape(scalearray, [9, 1])

        G1Train, G2Train, _ = self.exactSol(xBdryTrain, yBdryTrain)
        G1xTrain, G1yTrain, G2xTrain, G2yTrain, _, _ = self.exactDx(xBdryTrain, yBdryTrain)
        _, _, iTrain = self.exactSol(xCompatTrain, yCompatTrain)
        U1bdryExactTrain = {}
        U1bdryExactTrain["value"] = np.reshape(G1Train, [xBdryTrain.shape[0],])
        U1bdryExactTrain["x"] = np.reshape(G1xTrain, [xBdryTrain.shape[0],])
        U1bdryExactTrain["y"] = np.reshape(G1yTrain, [xBdryTrain.shape[0],])
        U2bdryExactTrain = {}
        U2bdryExactTrain["value"] = np.reshape(G2Train, [xBdryTrain.shape[0],])
        U2bdryExactTrain["x"] = np.reshape(G2xTrain, [xBdryTrain.shape[0],])
        U2bdryExactTrain["y"] = np.reshape(G2yTrain, [xBdryTrain.shape[0],])

        G1Val, G2Val, _ = self.exactSol(xBdryVal, yBdryVal)
        G1xVal, G1yVal, G2xVal, G2yVal, _, _ = self.exactDx(xBdryVal, yBdryVal)
        _, _, iVal = self.exactSol(xCompatVal, yCompatVal)
        U1bdryExactVal = {}
        U1bdryExactVal["value"] = np.reshape(G1Val, [xBdryVal.shape[0],])
        U1bdryExactVal["x"] = np.reshape(G1xVal, [xBdryVal.shape[0],])
        U1bdryExactVal["y"] = np.reshape(G1yVal, [xBdryVal.shape[0],])
        U2bdryExactVal = {}
        U2bdryExactVal["value"] = np.reshape(G2Val, [xBdryVal.shape[0],])
        U2bdryExactVal["x"] = np.reshape(G2xVal, [xBdryVal.shape[0],])
        U2bdryExactVal["y"] = np.reshape(G2yVal, [xBdryVal.shape[0],])
        
        DATA = {}
        DATA["F1"] = f1Train
        DATA["F2"] = f2Train
        DATA["F3"] = f3Train
        DATA["F3x"] = f3xTrain
        DATA["F3y"] = f3yTrain
        DATA["G1"] = G1Train
        DATA["G2"] = G2Train
        DATA["G1x"] = G1xTrain
        DATA["G1y"] = G1yTrain
        DATA["G2x"] = G2xTrain
        DATA["G2y"] = G2yTrain
        DATA["G1x"] = G1xTrain
        DATA["G1y"] = G1yTrain
        DATA["I"] = iTrain

        while (k < self.maxRef):
            self.iterFlag = k

            # width and scaling for kth basis function
            neuronsk = neuronsarray[k,0]
            scalek = scalearray[k,0]
            numWavesk = 3 + int(np.floor(k/2))
            sizesk = [2 * self.QuadRules.interior_x.shape[1] * numWavesk, neuronsk]

            # weights and biases
            params = self.initParams(sizesk, k, numWavesk)

            # learning rate
            lr0 = self.BASE_LR / (self.ITER_DECAY**k)
            lr0 = optimizers.exponential_decay(lr0, self.EXP_DECAY, self.maxEpoch)
            opt_init, opt_update, get_params = optimizers.adam(lr0)
            opt_state = opt_init(params)

            # Get the initial set of parameters
            params = get_params(opt_state)
            activationcoeffs = {}
            activationcoeffs_singular = {}

            # compute Energy and L2 error
            L2_k, Energy_k, L20, Energy0, L2p_k = self.computeError(uVal)

            gradCheck0 = 1.e5
            gradCheck1 = 1.e2

            i = 0

            # for i in range(NUM_EPOCHS):
            while (np.abs(gradCheck0 - gradCheck1)/np.abs(gradCheck0) > self.gradTol and i < self.maxEpoch):

                # update activation coefficients
                if (i%1 == 0):
                    (coeffsphi1reg, coeffsphi2reg, coeffsqreg, myCond) = self.galerkinUpdate(k, neuronsk, params, uTrain, scalek, DATA)

                if (self.isExtendedNetwork):
                    activationcoeffs["u1"] = np.reshape(coeffsphi1reg[0:sizesk[-1]], [sizesk[-1], 1])
                    activationcoeffs["u2"] = np.reshape(coeffsphi2reg[0:sizesk[-1]], [sizesk[-1], 1])
                    activationcoeffs["p"] = np.reshape(coeffsqreg[0:sizesk[-1]], [sizesk[-1], 1])
                    activationcoeffs_singular["u1"] = np.reshape(coeffsphi1reg[sizesk[-1]:], [2 * self.alphaSize + 2, 1])
                    activationcoeffs_singular["u2"] = np.reshape(coeffsphi2reg[sizesk[-1]:], [2 * self.alphaSize + 2, 1])
                    activationcoeffs_singular["p"] = np.reshape(coeffsqreg[sizesk[-1]:], [2 * self.alphaSize + 2, 1])
                else:
                    activationcoeffs["u1"] = coeffsphi1reg
                    activationcoeffs["u2"] = coeffsphi2reg
                    activationcoeffs["p"] = coeffsqreg

                # evaluate loss and approximate L2
                loss_i = self.computeLoss(params, k, xVal[:,0], yVal[:,0], self.QuadRulesVal.interior_w[:,0], 
                                          xBdryVal[:,0], yBdryVal[:,0], self.QuadRulesVal.boundary_w[:,0],
                                          xCompatVal[:,0], yCompatVal[:,0], self.QuadRulesVal.compatibility_w[:,0],
                                          self.QuadRulesVal.boundary_t_x[:,0], self.QuadRulesVal.boundary_t_y[:,0],
                                          activationcoeffs, activationcoeffs_singular, scalek, 
                                          uVal, f1Val[:,0], f2Val[:,0], f3Val[:,0], f3xVal[:,0], f3yVal[:,0], U1bdryExactVal, U2bdryExactVal, iVal[:,0])
                loss_i = jnp.squeeze(-loss_i) / jnp.squeeze(Energy0)

                # self.L2_epoch.append(L2_k)
                self.Loss_epoch.append(loss_i)
                # self.approxL2_epoch.append(approxL2_i)
                self.Energy_epoch.append(Energy_k)

                print("Iter: ", k, ", Epoch: ", i, ", Loss: ", loss_i, ", Energy: ", Energy_k, ", L2 velocity: ", L2_k, ", L2 pressure: ", L2p_k)

                # get current weights and biases
                params, opt_state, loss = self.update(params, k, activationcoeffs, activationcoeffs_singular, scalek, uTrain,
                                                      f1Train[:,0], f2Train[:,0], f3Train[:,0], f3xTrain[:,0], f3yTrain[:,0], U1bdryExactTrain, U2bdryExactTrain, iTrain[:,0],
                                                      opt_state, opt_update, get_params)
                params_sep = self.separateParams(params)
                params_phi1 = params_sep["u1"]
                _, _, _, zeta, xi, _, _, _ = params_phi1[0]
                self.xi_epoch.append(xi.flatten())
                self.zeta_epoch.append(zeta.flatten())    

                alpha = (np.pi - 2.0*np.arccos(1.0/np.sqrt(10.0))) / 2.0
                zetaFixedU = 1.0 + 4.2266/(2.0*alpha)
                xiFixedU = 2.17466/(2.0*alpha)

                print("Eigenvalue: ", zeta, xi, "Exact: ", zetaFixedU, xiFixedU)

                gradCheck0 = gradCheck1
                gradCheck1 = loss_i

                i += 1

            # last activation coefficient update
            (coeffsphi1reg, coeffsphi2reg, coeffsqreg, _) = self.galerkinUpdate(k, neuronsk, params, uTrain, scalek, DATA)
            if (self.isExtendedNetwork):
                activationcoeffs["u1"] = np.reshape(coeffsphi1reg[0:sizesk[-1]], [sizesk[-1], 1])
                activationcoeffs["u2"] = np.reshape(coeffsphi2reg[0:sizesk[-1]], [sizesk[-1], 1])
                activationcoeffs["p"] = np.reshape(coeffsqreg[0:sizesk[-1]], [sizesk[-1], 1])
                activationcoeffs_singular["u1"] = np.reshape(coeffsphi1reg[sizesk[-1]:], [2 * self.alphaSize + 2, 1])
                activationcoeffs_singular["u2"] = np.reshape(coeffsphi2reg[sizesk[-1]:], [2 * self.alphaSize + 2, 1])
                activationcoeffs_singular["p"] = np.reshape(coeffsqreg[sizesk[-1]:], [2 * self.alphaSize + 2, 1])
            else:
                activationcoeffs["u1"] = coeffsphi1reg
                activationcoeffs["u2"] = coeffsphi2reg
                activationcoeffs["p"] = coeffsqreg

            # evaluate loss and approximate L2
            loss_i = self.computeLoss(params, k, xVal[:,0], yVal[:,0], self.QuadRulesVal.interior_w[:,0], 
                                        xBdryVal[:,0], yBdryVal[:,0], self.QuadRulesVal.boundary_w[:,0],
                                        xCompatVal[:,0], yCompatVal[:,0], self.QuadRulesVal.compatibility_w[:,0],
                                        self.QuadRulesVal.boundary_t_x[:,0], self.QuadRulesVal.boundary_t_y[:,0],
                                        activationcoeffs, activationcoeffs_singular, scalek, 
                                        uVal, f1Val[:,0], f2Val[:,0], f3Val[:,0], f3xVal[:,0], f3yVal[:,0], U1bdryExactVal, U2bdryExactVal, iVal[:,0])
            loss_i = jnp.squeeze(-loss_i) / jnp.squeeze(Energy0)

            # append phi_i to basis
            self.appendBasis(k, params, activationcoeffs, activationcoeffs_singular, scalek)

            # error 
            u1eStream, u2eStream, peStream = self.exactSol(xStream, yStream)

            nstream = int(np.sqrt(xStream.shape[0]))
            xPlot = np.reshape(xStream, [nstream, nstream])
            yPlot = np.reshape(yStream, [nstream, nstream])

            ERROR_U1 = np.reshape(u1eStream - uStream["u1"]["interior_value"], [nstream, nstream])
            ERROR_U2 = np.reshape(u2eStream - uStream["u2"]["interior_value"], [nstream, nstream])
            ERROR_P = np.reshape(peStream - uStream["p"]["interior_value"], [nstream, nstream])
            BASIS_PHI1 = np.reshape(self.basis_stream["u1"][-1]["interior_value"], [nstream, nstream])
            BASIS_PHI2 = np.reshape(self.basis_stream["u2"][-1]["interior_value"], [nstream, nstream])
            BASIS_Q =np.reshape(self.basis_stream["p"][-1]["interior_value"], [nstream, nstream])

            myTitle = r"Approximate error $\varphi_{1,i}^{NN}$"
            myPath = self.RESULTS_PATH + "/basis_phi1_" + str(k) + ".png"
            self.plotSurface(k, xPlot, yPlot, BASIS_PHI1, myTitle, myPath)

            myTitle = r"Approximate error $\varphi_{2,i}^{NN}$"
            myPath = self.RESULTS_PATH + "/basis_phi2_" + str(k) + ".png"
            self.plotSurface(k, xPlot, yPlot, BASIS_PHI2, myTitle, myPath)

            myTitle = r"Approximate error $\varphi_{q,i}^{NN}$"
            myPath = self.RESULTS_PATH + "/basis_q_" + str(k) + ".png"
            self.plotSurface(k, xPlot, yPlot, BASIS_Q, myTitle, myPath)

            # approximate solution using basis functions
            (uTrain, uVal, uStream, coeffsbasisU1, coeffsbasisU2, coeffsbasisP) = self.galerkinSolve(k)

            myTitle = r"Approximation $u_{1,i}$"
            myPath = self.RESULTS_PATH + "/solution_u1_" + str(k) + ".png"
            self.plotSurface(k, xPlot, yPlot, np.reshape(uStream["u1"]["interior_value"], [nstream, nstream]), myTitle, myPath)

            myTitle = r"Approximation $u_{2,i}$"
            myPath = self.RESULTS_PATH + "/solution_u2_" + str(k) + ".png"
            self.plotSurface(k, xPlot, yPlot, np.reshape(uStream["u2"]["interior_value"], [nstream, nstream]), myTitle, myPath)

            myTitle = r"Approximation Velocity"
            myPath = self.RESULTS_PATH + "/solution_speed_" + str(k) + ".png"
            self.plotVelocity(k, xPlot, yPlot, np.reshape(np.sqrt(uStream["u1"]["interior_value"]**2 + uStream["u2"]["interior_value"]**2), [nstream, nstream]), myTitle, myPath)

            myTitle = r"Approximation $p_{i}$"
            myPath = self.RESULTS_PATH + "/solution_p_" + str(k) + ".png"
            self.plotSurface(k, xPlot, yPlot, np.reshape(uStream["p"]["interior_value"], [nstream, nstream]), myTitle, myPath)

            # save results
            coeffsU1 = [coeffsphi1reg, coeffsbasisU1]
            coeffsU2 = [coeffsphi2reg, coeffsbasisU2]
            coeffsP = [coeffsqreg, coeffsbasisP]

            trainedParams_k = optimizers.unpack_optimizer_state(opt_state)
            self.trainedParams.append([trainedParams_k, coeffsU1, coeffsU2, coeffsP, scalek])

            k += 1

            # update error indicator
            errorIndicator = loss_i

            # self.L2_epoch.append(L2_k)
            self.Loss_epoch.append(loss_i)
            # self.approxL2_epoch.append(approxL2_i)
            self.Energy_epoch.append(Energy_k)

            self.L2_iter.append(L2_k)
            self.L2p_iter.append(L2p_k)
            # self.L2weighted_iter.append(L2weighted_k)
            self.Loss_iter.append(loss_i)
            # self.approxL2_iter.append(approxL2_i)
            # self.approxL2weighted_iter.append(approxL2weighted_i)
            self.Energy_iter.append(np.squeeze(Energy_k))
            self.cond_iter.append(myCond)

            # plot loss per epoch and iteration
            fig = plt.figure()
            # plt.subplot(1,2,1)
            plt.semilogy(np.arange(1,len(self.basis["p"])+1), self.Energy_iter, 'o-', label=r'$|||u-u_{i-1}|||$')
            plt.semilogy(np.arange(1,len(self.basis["p"])+1), np.asarray(self.Loss_iter), 'o--', color='tab:orange', label=r'$\eta(u_{i-1},\varphi_{i}^{NN})$')
            # plt.semilogy(np.arange(1,len(self.basisq)+1), self.natEnergy_iter, '^-', color='tab:blue', label=r'$|||u-u_{i-1}|||_{0}$')
            plt.semilogy(np.arange(1,len(self.basis["p"])+1), self.L2_iter, 's-', fillstyle='none', color='tab:blue', label=r'$||u-u_{i-1}||_{L^{2}}$')
            plt.semilogy(np.arange(1,len(self.basis["p"])+1), self.L2p_iter, 'v-', fillstyle='none', color='tab:blue', label=r'$||p-p_{i-1}||_{L^{2}}$')
            plt.xlabel('Number of basis functions', fontsize=16)
            plt.ylabel('Error', fontsize=16)
            mystr = r'Error vs. Number of Basis Functions, $\beta=5/3$'
            plt.title(mystr, fontsize=16)
            plt.legend(fontsize=12)
            ratio = 0.75
            plt.grid()
            plt.savefig(self.RESULTS_PATH + "/GNN_error_%d" % k)
            plt.close()

            plt.figure()
            plt.semilogy(np.arange(1,len(self.Loss_epoch)+1), np.squeeze(self.Energy_epoch), '-', label=r'$|||u-u_{i-1}|||$')
            plt.semilogy(np.arange(1,len(self.Loss_epoch)+1), [L for L in self.Loss_epoch], '--', color='tab:orange', label='Loss')
            plt.xlabel('Training epoch', fontsize=16)
            plt.ylabel('Loss', fontsize=16)
            mystr = 'Loss vs. Training Epoch'
            plt.title(mystr, fontsize=16)
            plt.legend(fontsize=12)
            ratio = 0.75
            plt.grid()
            plt.savefig(self.RESULTS_PATH + "/epoch_error_%d" % k)
            plt.close()

            alpha = (np.pi - 2.0*np.arccos(1.0/np.sqrt(10.0))) / 2.0
            zetaFixedU = 1.0 + 4.2266/(2.0*alpha)
            xiFixedU = 2.17466/(2.0*alpha)

            plt.figure()
            plt.plot(self.zeta_epoch, '--', color='tab:blue', label="Approximate real part")
            plt.plot(self.xi_epoch, '--', color='tab:orange', label="Approximate imaginary part")
            plt.plot(zetaFixedU * np.ones([len(self.zeta_epoch),]), '-', color='tab:blue', label="Exact real part")
            plt.plot(xiFixedU * np.ones([len(self.zeta_epoch),]), '-', color='tab:orange', label="Exact imaginary part")
            plt.xlabel("Epoch")
            plt.ylabel("Eigenvalue")
            plt.legend()
            plt.savefig(self.RESULTS_PATH + "/eigenvalue_training.png")

        mdic = {"params": self.trainedParams,
                "loss_epoch": self.Loss_epoch,
                "energy_epoch": self.Energy_epoch,
                "zeta_epoch": self.zeta_epoch,
                "xi_epoch": self.xi_epoch,
                "L2_iter": self.L2_iter,
                "energy_iter": self.Energy_iter,
                "loss_iter": self.Loss_iter}
        pickle.dump(mdic, open(self.RESULTS_PATH + "/saved_stokes2D_eddylearning_results.pkl", "wb"))
  
