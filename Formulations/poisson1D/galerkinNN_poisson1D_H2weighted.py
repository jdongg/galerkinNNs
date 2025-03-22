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
from typing import Callable, List, Type
from functools import partial
jax.config.update("jax_enable_x64", True)
from matplotlib import ticker
from QuadratureRules.QuadratureRules import QuadratureRules

niceMathTextForm = ticker.ScalarFormatter(useMathText=True)

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

    which is always SPD. This class solves the Poisson equation using a least-sqaures variational formulation posed
    on H2:

                a(u,v) = (u'', v'')_L2(Ω) + C*(u, v)_L2(∂Ω)
                L(v)   = (f, -v'')_L2(Ω) + C*(u_D, v)_L2(∂Ω)

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
    def __init__(self, neurons: int, scale: np.float64, Lx: np.float64, isExtended: bool, 
                 QuadTrain: Type[QuadratureRules], QuadVal: Type[QuadratureRules], xStream: np.ndarray, f: Callable, 
                 u: Callable, Du: Callable, tolerance: np.float64, gradTolerance: np.float64, 
                 boundaryPen1: np.float64, boundaryPen2: np.float64, scale_increment: np.float64, base_lr: np.float64, 
                 iter_decay_rate: np.float64, exp_decay_rate: np.float64, maxEpoch: int, maxRef: int, 
                 RESULTS_PATH: str, beta: np.float64):

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

        # width of bounding rectangle for domain
        self.Lx = Lx

        # weighted formulation power
        self.beta = beta

        # generate quadrature rule; training
        self.QuadRules = QuadTrain

        # generate quadrature rule; validation
        self.QuadRulesVal = QuadVal

        # toggle extended network architecture
        self.isExtendedNetwork = isExtended
        self.alphaSize = 20

        # list of solution fields
        self.solution_fields = ["u"]

        # list of derivatives in the interior, spatial boundary, and temporal boundary
        # for each solution field
        #   0 - x
        # 
        self.intDerivatives = {}
        self.bdryDerivatives = {}

        self.intDerivatives["u"] = ["0", "00"] 
        self.bdryDerivatives["u"] = [] 

        # specify which solution fields have spatial boundary
        self.hasSpatialBoundary = {}
        self.hasSpatialBoundary["u"] = True

        self.xStream0 = xStream

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

        # results epochs
        self.L2_epoch = []
        self.Loss_epoch = []
        self.approxL2_epoch = []
        self.Energy_epoch = []
        self.theta_epoch = []

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


    def parseDerivativeString(self, der: str):
        der_print = ""
        for der_i in der:
            if (der_i == "0"):
                der_print += "x"
        return der_print 


    def boxInit(self, m: int, n: int, flag: int, ki: int, L: int, ell: int, numWaves: int):
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
                # print(k[i])
                for j in range(m):
                    W[i,j] = k[0,i]*nhat[i,j]

                b[i] = -k[0,i]*np.sum(nhat[i,:] * p[i,:])

            Wr = W.T
            W = W.T
            b = b.T

        return W, b, Wr


    def initLayer(self, m: int, n: int, L: int, ell: int, m_alpha: int, k: int, numWaves: int):
        W, b, Wr = self.boxInit(m, n, 0, k, L, ell, numWaves)
        scale = self.scale + self.SCALE_INCR*k

        m = jnp.arange(0, numWaves)
        wxm = jnp.pow(2.0, m) * jnp.pi / self.Lx

        if (self.isExtendedNetwork):
            alpha = np.linspace(2./3., 2./3. + 2./3. * (m_alpha - 1), m_alpha)
            return W, b, Wr, scale, wxm
        else:
            return W, b, Wr, scale, wxm
        return


    def initParams(self, sizes: List[int], k: int, numWaves: int):
        return [self.initLayer(m, n, len(sizes)-1, i+1, int(sizes[1]/2), k, numWaves) for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:]))]	


    @partial(jax.jit, static_argnums=(0,5,6,))
    def createNetwork(self, X, params, Cnn, scale, numWaves, alphaSize=None, Calpha=None):
        """
        Forward pass for creating feedforward DNN with Fourier feature mapping of input layer [1,2].
        An optional extended architecture is used [3] which supplements the DNN with singular functions
        based on eigenfunctions of the corresponding operator pencil.

        [1] Aldirany, Z., Cottereau, R., Laforest, M., & Prudhomme, S. (2024). Multi-level neural networks 
            for accurate solutions of boundary-value problems. Computer Methods in Applied Mechanics and Engineering, 419, 116666.
        [2] Tancik, M., Srinivasan, P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., ... & Ng, R. (2020). 
            Fourier features let networks learn high frequency functions in low dimensional domains. Advances in neural 
            information processing systems, 33, 7537-7547.
        [3] Ainsworth, M., & Dong, J. (2024). Extended Galerkin neural network approximation of singular variational 
            problems with error control. arXiv preprint arXiv:2405.00815.
        """
        # first hidden layer
        W, b, Wr, scale_NN, wxm = params[0]

        # Fourier feature map
        Z = jnp.column_stack((jnp.cos(jnp.outer(X, wxm)), jnp.sin(jnp.outer(X, wxm))))
        H = jnp.tanh((jnp.matmul(Z, W) + b))

        for idx, (W, b, Wr, scale_NN, _) in enumerate(params[1:]):
            if (idx == len(params)-2):
                H = jnp.tanh((jnp.matmul(H, W) + b)) #+ jnp.matmul(Hend, Wr)
            else:
                H = jnp.tanh((jnp.matmul(H, W) + b))
        Unn = H @ Cnn

        if (self.isExtendedNetwork):
            alpha = jnp.reshape(jnp.linspace(2./3., 2./3. * alphaSize, alphaSize), [1, alphaSize])
            Ualpha = (jnp.pow(X, alpha)) @ Calpha
            U = Unn + Ualpha
        else:
            U = Unn

        return jnp.squeeze(U)


    @partial(jax.jit, static_argnums=(0,4,5,))
    def networkArrayReg(self, X, params, scale, numWaves, alphaSize=None):
        """
        Function which returns an N x n array where the jth column represents jth
        component of the last hidden layer (consisting of n neurons) at N quadrature nodes.
        """
        W, b, Wr, scale_NN, wxm, = params[0]

        # m = jnp.arange(0, numWaves)
        # wxm = jnp.pow(2.0, m) * jnp.pi / self.Lx
        # wym = jnp.pow(2.0, m) * jnp.pi / self.Ly
        Z = jnp.column_stack((jnp.cos(jnp.outer(X, wxm)), jnp.sin(jnp.outer(X, wxm))))
        Hnn = jnp.tanh((jnp.matmul(Z, W) + b))

        for idx, (W, b, Wr, scale_NN, _) in enumerate(params[1:]):
            if (idx == len(params)-2):
                Hnn = jnp.tanh((jnp.matmul(Hnn, W) + b)) #+ jnp.matmul(Hend, Wr)
            else:
                Hnn = jnp.tanh((jnp.matmul(Hnn, W) + b))

        if (self.isExtendedNetwork):
            alpha = jnp.reshape(jnp.linspace(2./3., 2./3. * alphaSize, alphaSize), [1, alphaSize])
            Hnn = jnp.column_stack((Hnn, jnp.power(X, alpha)))

        return Hnn


    def activMatrix(self, i, k, NN, solution0,
                     A, F, DATA):
        """
        Routine for assembling the linear system corresponding to 

                u ∈ Φ : a(u,v) = L(v) - a(u0,v)    ∀v ∈ Φ,      (*)

        where Φ = span{Ψ_i} and Ψ_i (i=1,...,n) are the n components of the last hidden layer of the DNN
        (plus potential singular functions if using the extended architecture). The discrete problem (*)
        performs a projection of the error u-u0 onto the space Φ and is a least squares training routine for
        the linear coefficients of the activation layer. More details can be found in [1], which is based
        on the strong-form least squares training introduced in [2].

        [1] Ainsworth, M., & Dong, J. (2021). Galerkin neural networks: A framework for approximating 
            variational equations with error control. SIAM Journal on Scientific Computing, 43(4), A2474-A2501.
        [2] Cyr, E. C., Gulian, M. A., Patel, R. G., Perego, M., & Trask, N. A. (2020, August). 
            Robust training and initialization of deep neural networks: An adaptive basis viewpoint. 
            In Mathematical and Scientific Machine Learning (pp. 512-536). PMLR.
        """
        xTrain = np.reshape(self.QuadRules.interior_x[:,0], [self.QuadRules.interior_x.shape[0], 1])

        if (self.isExtendedNetwork):
            w = np.sqrt(xTrain**2)**self.beta
            wBdry = 1.0
        else:
            w = 1.0
            wBdry = 1.0

        # interior terms
        # A matrix
        NNdx2phi_idx = np.reshape(NN["u"]["interior_xx"][:,i], [self.QuadRules.interior_x.shape[0], 1])
        NNdx2phi_jdx = NN["u"]["interior_xx"]

        A[i,:] = np.sum(self.QuadRules.interior_w * (w*(NNdx2phi_idx)) * (w*(NNdx2phi_jdx)), axis=0)

        # boundary terms
        NNbdryphi_idx = np.reshape(NN["u"]["boundary_value"][:,i], [self.QuadRules.boundary_x.shape[0], 1])
        NNbdryphi_jdx = NN["u"]["boundary_value"]

        A[i,:] += self.boundaryPen1 * np.sum(self.QuadRules.boundary_w * (wBdry*NNbdryphi_idx) * (wBdry*NNbdryphi_jdx), axis=0)
        
        # RHS
        F[i] = np.sum(self.QuadRules.interior_w * (w*DATA["F1"]) * (w*(-NNdx2phi_idx)))
        F[i] += self.boundaryPen1 * np.sum(self.QuadRules.boundary_w * (wBdry*NNbdryphi_idx) * (wBdry*DATA["G1"]))
        F[i] += -np.sum(self.QuadRules.interior_w * (w*(NNdx2phi_idx)) * (w*(solution0["u"]["interior_xx"])))
        F[i] += -self.boundaryPen1 * np.sum(self.QuadRules.boundary_w * (wBdry*NNbdryphi_idx) * (wBdry*solution0["u"]["boundary_value"]))

        return


    def galerkinUpdate(self, k, neurons, params, solution0, scale, numWaves, alphaSize, DATA):
        """
        Routine which computes the subspace Φ = span{Ψ_i}, where Ψ_i (i=1,...,n) are the n components 
        of the last hidden layer of the DNN (plus potential singular functions if using the extended architecture).
        The components Ψ_i and its derivatives in the interior of the domain and on its boundary 
        are computed and stored in the dictionary NNoutput. 

        Returns the linear coefficients of the (linear) activation layer of the DNN by performing
        Galerkin orthogonal projection of the error u-solution0 onto Φ.
        """
        params_i = {}
        params_i["u"] = params

        N = self.QuadRules.interior_x.shape[0]
        if (self.isExtendedNetwork):
            dim = neurons + alphaSize 
        else:
            dim = neurons
        A = np.zeros([dim, dim])
        F= np.zeros([dim, 1])
  
        xTrain = np.reshape(self.QuadRules.interior_x[:,0], [self.QuadRules.interior_x.shape[0], 1])
        xBdryTrain = np.reshape(self.QuadRules.boundary_x[:,0], [self.QuadRules.boundary_x.shape[0], 1])

        # compute the derivatives as needed in the interior
        NNoutput = {}
        for field in self.solution_fields:
            NNoutput[field] = {}

            if (self.isExtendedNetwork):
                opt_args = alphaSize
            else:
                opt_args = None

            # value of the basis function
            NNoutput[field]["interior_value"] = jnp.squeeze(jax.vmap(self.networkArrayReg, (0, None, None, None, None),  0)
                                                            (xTrain, params_i[field], scale, numWaves, opt_args))
            for der in self.intDerivatives[field]:	
                # compute the requested derivative
                der_iter = 0
                for der_i in der:
                    if (der_iter == 0):
                        grad_i = jax.vmap(jax.jacfwd(self.networkArrayReg, int(der_i)), (0, None, None, None, None), 0)
                    else:
                        grad_i = jax.vmap(jax.jacfwd(grad_i, int(der_i)), (0, None, None, None, None), 0)
                    der_iter += 1
                der_print = self.parseDerivativeString(der)
                # value of its derivatives
                NNoutput[field]["interior_" + der_print] = jnp.squeeze(grad_i(xTrain, params_i[field], scale, numWaves, opt_args))

            # compute corresponding derivatives on the spatial boundary
            if self.hasSpatialBoundary[field]:
                NNoutput[field]["boundary_value"] = jnp.squeeze(jax.vmap(self.networkArrayReg, (0, None, None, None, None),  0)
                                                                (xBdryTrain, params_i[field], scale, numWaves, opt_args))
    
            for der in self.bdryDerivatives[field]:
                # compute the requested derivative
                der_iter = 0
                for der_i in der:
                    if (der_iter == 0):
                        grad_i = jax.vmap(jax.jacfwd(self.networkArrayReg, int(der_i)), (0, None, None, None, None), 0)
                    else:
                        grad_i = jax.vmap(jax.jacfwd(grad_i, int(der_i)), (0, None, None, None, None), 0)
                    der_iter += 1
                der_print = self.parseDerivativeString(der)
                NNoutput[field]["boundary_" + der_print] = jnp.squeeze(grad_i(xBdryTrain, params_i[field], scale, numWaves, opt_args))

        # assemble matrices
        t0 = time.time()
        Parallel(n_jobs=8, backend="threading")(
            delayed(self.activMatrix)(i, k, NNoutput, solution0,
                                      A, F, DATA) for i in range(dim))
        t1 = time.time()
  
        c, _, _, _ = scipy.linalg.lstsq(A, F, cond=None, lapack_driver='gelsy', check_finite=False)
        # c, _, _, _ = scipy.linalg.lstsq(K, F, cond=None)
        cphi = np.reshape(c, [dim,1])

        myCond = np.linalg.cond(A)
        print("Time elapsed for assembly: ", t1-t0)

        return cphi, myCond


    @partial(jax.jit, static_argnums=(0,2,10,11,))
    def computeLoss(self, params, k, 
                    X, W, 
                    Xbdry, Wbdry, 
                    activationcoeffs, activationcoeffs_singular,
                    scalek, numWaves, alphaSize, SOL0, F1, U0bdryexact):
        """
        Evaluates the loss function which is given by

                Loss[v] = -|<r(u0), v>| / a(v,v)^{1/2}.

        The minimum of the loss function (when v is taken over the infinite-dimensional space X) is
        the error u-u0. The minimizer φ of the loss is used as a basis function for a finite-dimensional
        subspace S_i := span{u0, φ1, ..., φn}. The basis functions are corrections to the initial coarse 
        approximation u0.
        """
        params_i = {}
        params_i["u"] = params

        # compute required interior fields
        phi = {}
        for field in self.solution_fields:
            phi[field] = {}

            opt_args = {}
            if (self.isExtendedNetwork):
                opt_args["coeffs"] = activationcoeffs_singular[field]
                opt_args["alphaSize"] = alphaSize
            else:
                opt_args["coeffs"] = None
                opt_args["alphaSize"] = None

            phi[field]["interior_value"] = self.createNetwork(X, params_i[field], activationcoeffs[field], scalek, 
                                                              numWaves, opt_args["alphaSize"], opt_args["coeffs"])
            for der in self.intDerivatives[field]:
                der_iter = 0
                for der_i in der:
                    if (der_iter == 0):
                        grad_i = jax.grad(self.createNetwork, int(der_i))
                    else:
                        grad_i = jax.grad(grad_i, int(der_i))
                    der_iter += 1
                der_print = self.parseDerivativeString(der)
                phi[field]["interior_" + der_print] = (jax.vmap(grad_i, (0, None, None, None, None, None, None))
                                                       (X, params_i[field], activationcoeffs[field], scalek, numWaves, opt_args["alphaSize"], opt_args["coeffs"]))

            # compute required boundary fields 
            if self.hasSpatialBoundary[field]:
                phi[field]["boundary_value"] = self.createNetwork(Xbdry, params_i[field], activationcoeffs[field], scalek, 
                                                                  numWaves, opt_args["alphaSize"], opt_args["coeffs"])

            for der in self.bdryDerivatives[field]:
                der_iter = 0
                for der_i in der:
                    if (der_iter == 0):
                        grad_i = jax.grad(self.createNetwork, int(der_i))
                    else:
                        grad_i = jax.grad(grad_i, int(der_i))
                    der_iter += 1
                der_print = self.parseDerivativeString(der)
                phi[field]["boundary_" + der_print] = (jax.vmap(grad_i, (0, None, None, None, None, None, None))
                                                       (Xbdry, params_i[field], activationcoeffs[field], scalek, numWaves, opt_args["alphaSize"], opt_args["coeffs"]))

        # compute loss function
        if (self.isExtendedNetwork):
            resWeight = jnp.power(jnp.sqrt(jnp.pow(X, 2)), self.beta)
            resWeightBdry = 1.0
        else:
            resWeight = 1.0
            resWeightBdry = 1.0

        # compute the energy norm sqrt(a(phi, phi))
        normint1 = jnp.sum(jnp.multiply(W, jnp.square(jnp.multiply(resWeight, phi["u"]["interior_xx"]))))
        normbdry1 = jnp.multiply(self.boundaryPen1, jnp.sum(jnp.multiply(Wbdry, jnp.square(resWeightBdry*phi["u"]["boundary_value"]))))
        norm = jnp.sqrt(normint1 + normbdry1)
        
        # compute the value of a(u0,phi)
        r1 = jnp.sum(jnp.multiply(W, jnp.multiply(jnp.multiply(resWeight, SOL0["u"]["interior_xx"][:,0]), 
                                                  jnp.multiply(resWeight, phi["u"]["interior_xx"]))))
        r2 = jnp.multiply(self.boundaryPen1, jnp.sum(jnp.multiply(Wbdry, jnp.multiply(resWeightBdry*SOL0["u"]["boundary_value"][:,0], resWeightBdry*phi["u"]["boundary_value"]))))

        # compute the value of L(phi)
        r3 = jnp.sum(jnp.multiply(W, jnp.multiply(jnp.multiply(resWeight, F1), jnp.multiply(resWeight, -phi["u"]["interior_xx"]))))
        r4 = jnp.multiply(self.boundaryPen1, jnp.sum(jnp.multiply(Wbdry, jnp.multiply(resWeightBdry*U0bdryexact["value"], resWeightBdry*phi["u"]["boundary_value"]))))
        res = r3+r4 - (r1+r2)

        Loss = jnp.multiply(-1.0, jnp.abs(jnp.divide(res, norm)))

        return jnp.squeeze(Loss)

    def computeError(self, solution0):
        """
        Computes the errors in the energy norm (always exactly computable even if closed form of solution is unknown)
        and L2 norm (assuming closed form of solution is known).
        """
        # compute Energy and L2 error
        xVal = np.reshape(self.QuadRulesVal.interior_x[:,0], [self.QuadRulesVal.interior_x.shape[0], 1])
        f = self.sourceFunc(xVal)

        xBdryVal = np.reshape(self.QuadRulesVal.boundary_x[:,0], [self.QuadRulesVal.boundary_x.shape[0], 1])

        if (self.isExtendedNetwork):
            resWeight = np.sqrt(xVal**2)**self.beta
            resWeightBdry = 1.0
        else:
            resWeight = 1.0
            resWeightBdry = 1.0

        uexact = self.exactSol(xVal)
        L2_k = np.sqrt(np.sum(self.QuadRulesVal.interior_w * ((solution0["u"]["interior_value"] - uexact)**2)))
        L2_k0 = np.sqrt(np.sum(self.QuadRulesVal.interior_w * (uexact**2)))
        L2_k = L2_k/L2_k0

        ubdryexact = self.exactSol(xBdryVal)
        uxbdryexact = self.exactDx(xBdryVal)

        exact1 = f
        approx1 = -solution0["u"]["interior_xx"]

        Energy_k = np.sum(self.QuadRulesVal.interior_w * (resWeight*(exact1 - approx1))**2)
        Energy_k += self.boundaryPen1 * np.sum(self.QuadRulesVal.boundary_w * (resWeightBdry*(solution0["u"]["boundary_value"] - ubdryexact))**2)
        Energy_k = np.sqrt(Energy_k)

        Energy0 = np.sum(self.QuadRulesVal.interior_w * (resWeight*(exact1))**2)
        Energy0 += self.boundaryPen1 * np.sum(self.QuadRulesVal.boundary_w * (resWeightBdry*ubdryexact)**2)
        Energy0 = np.sqrt(Energy0)
        Energy_k = Energy_k/Energy0

        return L2_k, Energy_k, L2_k0, Energy0


    def appendBasis(self, k, params, activationcoeffs, activationcoeffs_singular, scale, numWaves, alphaSize):
        """
        Adds the basis function described by params, activationcoeffs, and activationcoeffs_singular to the
        subspace. The basis function and its necessary derivatives are evaluate at the quadrature nodes and
        plotting points.
        """
        params_i = {}
        params_i["u"] = params

        xTrain = np.reshape(self.QuadRules.interior_x[:,0], [self.QuadRules.interior_x.shape[0], 1])
        xBdryTrain = np.reshape(self.QuadRules.boundary_x[:,0], [self.QuadRules.boundary_x.shape[0], 1])

        xVal = np.reshape(self.QuadRulesVal.interior_x[:,0], [self.QuadRulesVal.interior_x.shape[0], 1])
        xBdryVal = np.reshape(self.QuadRulesVal.boundary_x[:,0], [self.QuadRulesVal.boundary_x.shape[0], 1])

        xStream = np.reshape(self.xStream0[:,0], [self.xStream0.shape[0], 1])

        # basis evaluated at training points
        for field in self.solution_fields:
            field_basis = {}

            opt_args = {}
            if (self.isExtendedNetwork):
                opt_args["coeffs"] = activationcoeffs_singular[field]
                opt_args["alphaSize"] = alphaSize
            else:
                opt_args["coeffs"] = None
                opt_args["alphaSize"] = None
                
            # interior value and derivatives
            field_basis["interior_value"] = self.createNetwork(xTrain, params_i[field], activationcoeffs[field], scale, 
                                                               numWaves, opt_args["alphaSize"], opt_args["coeffs"])
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
                field_basis["interior_" + der_print] = (jax.vmap(grad_i, (0, None, None, None, None, None, None))
                                                        (xTrain[:,0], params_i[field], activationcoeffs[field], scale, numWaves, opt_args["alphaSize"], opt_args["coeffs"]))

            # spatial boundary value and derivatives
            if self.hasSpatialBoundary[field]:
                field_basis["boundary_value"] = self.createNetwork(xBdryTrain, params_i[field], activationcoeffs[field], scale, 
                                                                   numWaves, opt_args["alphaSize"], opt_args["coeffs"])
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
                field_basis["boundary_" + der_print] = (jax.vmap(grad_i, (0, None, None, None, None, None, None))
                                                        (xBdryTrain[:,0], params_i[field], activationcoeffs[field], scale, numWaves, opt_args["alphaSize"], opt_args["coeffs"]))

            self.basis[field].append(field_basis)


        # basis evaluated at validation points
        for field in self.solution_fields:
            field_basis = {}

            opt_args = {}
            if (self.isExtendedNetwork):
                opt_args["coeffs"] = activationcoeffs_singular[field]
                opt_args["alphaSize"] = alphaSize
            else:
                opt_args["coeffs"] = None
                opt_args["alphaSize"] = None

            # interior value and derivatives
            field_basis["interior_value"] = self.createNetwork(xVal, params_i[field], activationcoeffs[field], scale, 
                                                               numWaves, opt_args["alphaSize"], opt_args["coeffs"])
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
                field_basis["interior_" + der_print] = (jax.vmap(grad_i, (0, None, None, None, None, None, None))
                                                        (xVal[:,0], params_i[field], activationcoeffs[field], scale, numWaves, opt_args["alphaSize"], opt_args["coeffs"]))

            # spatial boundary value and derivatives
            if self.hasSpatialBoundary[field]:
                field_basis["boundary_value"] = self.createNetwork(xBdryVal, params_i[field], activationcoeffs[field], scale, 
                                                                   numWaves, opt_args["alphaSize"], opt_args["coeffs"])
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
                field_basis["boundary_" + der_print] = (jax.vmap(grad_i, (0, None, None, None, None, None, None))
                                                        (xBdryVal[:,0], params_i[field], activationcoeffs[field], scale, numWaves, opt_args["alphaSize"], opt_args["coeffs"]))

            self.basis_val[field].append(field_basis)


        # and evaluate at the equally-spaced plotting points
        for field in self.solution_fields:
            field_basis = {}

            opt_args = {}
            if (self.isExtendedNetwork):
                opt_args["coeffs"] = activationcoeffs_singular[field]
                opt_args["alphaSize"] = alphaSize
            else:
                opt_args["coeffs"] = None
                opt_args["alphaSize"] = None

            field_basis["interior_value"] = self.createNetwork(xStream, params_i[field], activationcoeffs[field], scale, 
                                                               numWaves, opt_args["alphaSize"], opt_args["coeffs"])
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
        xBdryTrain = np.reshape(self.QuadRules.boundary_x[:,0], [self.QuadRules.boundary_x.shape[0], 1])

        xVal = np.reshape(self.QuadRulesVal.interior_x[:,0], [self.QuadRulesVal.interior_x.shape[0], 1])
        xBdryVal = np.reshape(self.QuadRulesVal.boundary_x[:,0], [self.QuadRulesVal.boundary_x.shape[0], 1])

        xStream = np.reshape(self.xStream0[:,0], [self.xStream0.shape[0], 1])

        f1Train = self.sourceFunc(xTrain)
        G1  = self.exactSol(xBdryTrain)
        G1x, G1y = self.exactDx(xBdryTrain)

        f1Train = np.squeeze(f1Train)
        G1 = np.squeeze(G1)
        G1x = np.squeeze(G1x)
        G1y = np.squeeze(G1y)

        A = np.zeros([k+1, k+1])
        F = np.zeros([k+1, 1])

        if (self.isExtendedNetwork):
            w = np.squeeze(np.sqrt(xTrain**2)**self.beta)
            wBdry = 1.0
        else:
            w = 1.0
            wBdry = 1.0

        # the sums here are not vectorized so all arrays need to be one-dimensional. the basis functions are already
        # one-dimensional arrays. quadrature rules and exact functions for the RHS need to be converted to 1d
        wGlobalFlattened = np.squeeze(self.QuadRules.interior_w)
        wBdryFlattened = np.squeeze(self.QuadRules.boundary_w)

        # assemble linear system
        for idx in range(k+1):
            for jdx in range(k+1):
                A[idx,jdx] = np.sum(wGlobalFlattened * (w*(self.basis["u"][idx]["interior_xx"])) * 
                                                       (w*(self.basis["u"][jdx]["interior_xx"])))
                A[idx,jdx] += self.boundaryPen1 * np.sum(wBdryFlattened * (wBdry*self.basis["u"][idx]["boundary_value"]) * (wBdry*self.basis["u"][jdx]["boundary_value"]))
            F[idx] = np.sum(wGlobalFlattened * (w*(-self.basis["u"][idx]["interior_xx"])) * 
                                                 (w*f1Train))
            F[idx] += self.boundaryPen1 * np.sum(wBdryFlattened * (wBdry*self.basis["u"][idx]["boundary_value"]) * (wBdry*G1))

        # if MATLAB engine is installed, use it
        # c = eng.lsqminnorm(matlab.double(A.tolist()), matlab.double(b.tolist()))
        [c, _, _, _] = scipy.linalg.lstsq(A, F)
        c = np.asarray(c)
        cu = np.reshape(c, [k+1,1])

        c = {}
        c["u"] = cu

        print(cu)

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

        # repeat for equally-spaced plotting points, but only store the solution values
        uStream = {}
  
        for field in self.solution_fields:
            uStream[field] = {}
            val = np.zeros([xStream.shape[0], ])
            for idx in range(k+1):
                val += c[field][idx] * self.basis_stream[field][idx]["interior_value"]
            uStream[field]["interior_value"] = np.reshape(val, xStream.shape)
   
        return (uTrain, uVal, uStream, cu)


    def plotSurface(self, k, x, y, myTitle, myPath):

        fig = plt.figure()
        plt.plot(x, y)
        plt.xlabel('$x$', fontsize=16)
        plt.ylabel('$t$', fontsize=16)
        plt.title(myTitle, fontsize=16)
        plt.savefig(myPath)
        plt.close()

        return


    def update(self, params, k, activationcoeffs, activationcoeffs_singular,
               scalek, numWaves, alphaSize, SOL0, F1, U0bdryexact, 
               opt_state, opt_update, get_params):
        value, grads = jax.value_and_grad(self.computeLoss, argnums=0)(params, k, self.QuadRules.interior_x[:,0], self.QuadRules.interior_w[:,0], 
                                                                    self.QuadRules.boundary_x[:,0], self.QuadRules.boundary_w[:,0], 
                                                                    activationcoeffs, activationcoeffs_singular, scalek, numWaves, alphaSize,
                                                                    SOL0, F1, U0bdryexact)
        opt_state = opt_update(0, grads, opt_state)

        return get_params(opt_state), opt_state, value


    def generateBasis(self):
        """
        Generates a finite-dimensional subspace of dimension maxRefs and solves the variational problem.
        """
        xTrain = np.reshape(self.QuadRules.interior_x[:,0], [self.QuadRules.interior_x.shape[0], 1])
        xVal = np.reshape(self.QuadRulesVal.interior_x[:,0], [self.QuadRulesVal.interior_x.shape[0], 1])

        xBdryTrain = np.reshape(self.QuadRules.boundary_x[:,0], [self.QuadRules.boundary_x.shape[0], 1])
        xBdryVal = np.reshape(self.QuadRulesVal.boundary_x[:,0], [self.QuadRulesVal.boundary_x.shape[0], 1])

        f1Train = self.sourceFunc(xTrain)
        f1Val = self.sourceFunc(xVal)

        xStream = np.reshape(self.xStream0[:,0], [self.xStream0.shape[0], 1])

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

        # repeating for plotting points
        uStream = {}

        for field in self.solution_fields:
            uStream[field] = {}
            uStream[field]["interior_value"] = np.zeros([self.xStream0.shape[0],1])


        # generate new basis
        errorIndicator = 1.e3
        k = 0

        karray = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        neuronsarray = [int(self.neurons * (1.8**ki)) for ki in karray]
        neuronsarray = np.reshape(neuronsarray, [9, 1])

        scalearray = [np.minimum(self.scale + self.SCALE_INCR*(ki**1.1), 950.0) for ki in karray]
        scalearray = np.reshape(scalearray, [9, 1])

        G1Train = self.exactSol(xBdryTrain)
        UbdryExactTrain = {}
        UbdryExactTrain["value"] = np.reshape(G1Train, [xBdryTrain.shape[0],])

        G1Val = self.exactSol(xBdryVal)
        UbdryExactVal = {}
        UbdryExactVal["value"] = np.reshape(G1Val, [xBdryVal.shape[0],])
        
        DATA = {}
        DATA["F1"] = f1Train
        DATA["G1"] = G1Train

        while (k < self.maxRef):
            # width and scaling for kth basis function
            neuronsk = neuronsarray[k,0]
            scalek = scalearray[k,0]
            numWavesk = 3 + int(np.floor(k/2))
            sizesk = [2 * self.QuadRules.interior_x.shape[1] * numWavesk, neuronsk, neuronsk]
            alphaSizek = np.minimum(int(neuronsk/2), 10)

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
            L2_k, Energy_k, L20, Energy0 = self.computeError(uVal)

            gradCheck0 = 1.e5
            gradCheck1 = 1.e2

            i = 0
            while (np.abs(gradCheck0 - gradCheck1)/np.abs(gradCheck0) > self.gradTol and i < self.maxEpoch):

                # update activation coefficients
                kwargs = {}
                if (i%1 == 0):
                    (coeffsphireg, myCond) = self.galerkinUpdate(k, neuronsk, params, uTrain, scalek, numWavesk, alphaSizek, DATA)

                if (self.isExtendedNetwork):
                    activationcoeffs["u"] = np.reshape(coeffsphireg[0:sizesk[-1]], [sizesk[-1], 1])
                    activationcoeffs_singular["u"] = np.reshape(coeffsphireg[sizesk[-1]:], [alphaSizek, 1])
                    # print(activationcoeffs_singular["u"][0,0])
                else:
                    activationcoeffs["u"] = coeffsphireg

                # evaluate loss and approximate L2
                loss_i = self.computeLoss(params, k, xVal[:,0], self.QuadRulesVal.interior_w[:,0], 
                                          xBdryVal[:,0], self.QuadRulesVal.boundary_w[:,0],
                                          activationcoeffs, activationcoeffs_singular, scalek, numWavesk, alphaSizek,
                                          uVal, f1Val[:,0], UbdryExactVal)
                loss_i = jnp.squeeze(-loss_i) / jnp.squeeze(Energy0)

                # self.L2_epoch.append(L2_k)
                self.Loss_epoch.append(loss_i)
                # self.approxL2_epoch.append(approxL2_i)
                self.Energy_epoch.append(Energy_k)

                _, _, _, scale_trained, wxm = params[-1]

                print("Iter: ", k, ", Epoch: ", i, ", Loss: ", loss_i, ", Energy: ", Energy_k, ", L2 velocity: ", L2_k, ", Waves: ", wxm)

                # get current weights and biases
                params, opt_state, loss = self.update(params, k, activationcoeffs, activationcoeffs_singular,
                                                      scalek, numWavesk, alphaSizek, uTrain, f1Train[:,0], UbdryExactTrain,
                                                      opt_state, opt_update, get_params)
                # self.paramUpperBound(weights_eval, biases_eval, coeffs, phi, phixx, phibdry, feed_dict_train)

                gradCheck0 = gradCheck1
                gradCheck1 = loss_i

                i += 1

            # last activation coefficient update
            (coeffsphireg, _) = self.galerkinUpdate(k, neuronsk, params, uTrain, scalek, numWavesk, alphaSizek, DATA)
            if (self.isExtendedNetwork):
                activationcoeffs["u"] = np.reshape(coeffsphireg[0:sizesk[-1]], [sizesk[-1], 1])
                activationcoeffs_singular["u"] = np.reshape(coeffsphireg[sizesk[-1]:], [alphaSizek, 1])
            else:
                activationcoeffs["u"] = coeffsphireg

            # evaluate loss and approximate L2
            loss_i = self.computeLoss(params, k, xVal[:,0], self.QuadRulesVal.interior_w[:,0], 
                                        xBdryVal[:,0], self.QuadRulesVal.boundary_w[:,0],
                                        activationcoeffs, activationcoeffs_singular, scalek, numWavesk, alphaSizek,
                                        uVal, f1Val[:,0], UbdryExactVal)
            loss_i = jnp.squeeze(-loss_i) / jnp.squeeze(Energy0)

            # append phi_i to basis
            self.appendBasis(k, params, activationcoeffs, activationcoeffs_singular, scalek, numWavesk, alphaSizek)
    
            # error 
            ueStream = self.exactSol(xStream)

            ERROR_U = ueStream - np.reshape(uStream["u"]["interior_value"], [xStream.shape[0],1])
            BASIS_PHI = self.basis_stream["u"][-1]["interior_value"]

            myTitle = r"Exact error $u-u_{i-1}$"
            myPath = self.RESULTS_PATH + "/exact_error_" + str(k) + ".png"
            self.plotSurface(k, xStream, ERROR_U, myTitle, myPath)

            myTitle = r"Approximate error $\varphi_{i}^{NN}$"
            myPath = self.RESULTS_PATH + "/basis_" + str(k) + ".png"
            self.plotSurface(k, xStream, np.reshape(BASIS_PHI, [xStream.shape[0],1]), myTitle, myPath)

            # approximate solution using basis functions
            (uTrain, uVal, uStream, coeffsbasisU) = self.galerkinSolve(k)

            myTitle = r"Approximation $u_{i}$"
            myPath = self.RESULTS_PATH + "/solution_" + str(k) + ".png"
            self.plotSurface(k, xStream, np.reshape(uStream["u"]["interior_value"], [xStream.shape[0],1]), myTitle, myPath)

            # save results
            coeffsU = [coeffsphireg, coeffsbasisU]

            trainedParams_k = optimizers.unpack_optimizer_state(opt_state)
            self.trainedParams.append([params, coeffsU, scalek])

            k += 1

            # update error indicator
            errorIndicator = loss_i

            # self.L2_epoch.append(L2_k)
            self.Loss_epoch.append(loss_i)
            # self.approxL2_epoch.append(approxL2_i)
            self.Energy_epoch.append(Energy_k)

            # self.L2_iter.append(L2_k)
            # self.L2weighted_iter.append(L2weighted_k)
            self.Loss_iter.append(loss_i)
            # self.approxL2_iter.append(approxL2_i)
            # self.approxL2weighted_iter.append(approxL2weighted_i)

            L2_k, Energy_k, L20, Energy0 = self.computeError(uVal)
            self.Energy_iter.append(np.squeeze(Energy_k))
            self.L2_iter.append(np.squeeze(L2_k))
            self.cond_iter.append(myCond)

            # plot loss per epoch and iteration
            fig = plt.figure()
            # plt.subplot(1,2,1)
            plt.semilogy(np.arange(1,len(self.basis["u"])+2), [1.0] + self.Energy_iter, 'o-', label=r'$|||u-u_{i-1}|||$')
            plt.semilogy(np.arange(1,len(self.basis["u"])+1), np.asarray(self.Loss_iter), 'o--', color='tab:orange', label=r'$\eta(u_{i-1},\varphi_{i}^{NN})$')
            plt.semilogy(np.arange(1,len(self.basis["u"])+2), [1.0] + self.L2_iter, 's-', fillstyle='none', color='tab:blue', label=r'$||u-u_{i-1}||_{L^{2}}$')
            plt.xlabel('Number of basis functions', fontsize=16)
            plt.ylabel('Error', fontsize=16)
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

        mdic = {"params": self.trainedParams,
                "loss_epoch": self.Loss_epoch,
                "energy_epoch": self.Energy_epoch,
                "L2_iter": self.L2_iter,
                "energy_iter": self.Energy_iter,
                "loss_iter": self.Loss_iter}
        pickle.dump(mdic, open(self.RESULTS_PATH + "/saved_poisson1d_results.pkl", "wb"))
  
        return mdic 