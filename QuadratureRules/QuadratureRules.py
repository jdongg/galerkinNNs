import numpy as np
from scipy.special import roots_jacobi, eval_jacobi

class QuadratureRules:
    def __init__(self):
        self.interior_x = None
        self.interior_w = None
        self.boundary_x = None
        self.boundary_w = None
        
        # only for 2D problems
        self.boundary_t_x = None    # x-component of tangent vector
        self.boundary_t_y = None    # y-component of tangent vector
        self.boundary_n_x = None    # x-component of normal vector
        self.boundary_n_y = None    # y-component of normal vector

        # only for Stokes
        self.compatibility_x = None 
        self.compatibility_w = None 

        # only for 3D problems 
        self.boundary_t_z = None    # z-component of tangent vector
        self.boundary_n_z = None    # z-component of normal vector

        # only for time-dependent problems
        self.boundary_t_t = None

    def GaussLobatto1D(self, ng: int):
        x = np.zeros([ng,])
        w = np.zeros([ng,])

        x[0] = -1.0 
        x[-1] = 1.0
        w[0] = 2.0 / (ng * (ng - 1.0))
        w[-1] = 2.0 / (ng * (ng - 1.0))

        # interior nodes are roots of d/dx[P^(0,0)_{n-1}], which are also
        # the first n-1 roots of P^(1,1)_{n-2}.
        xint, _ = roots_jacobi(ng - 2, 1.0, 1.0)
        x[1:-1] = np.squeeze(xint)
        w[1:-1] = np.squeeze(2.0 / (ng * (ng - 1.0) * eval_jacobi(ng - 1, 0.0, 0.0, xint)**2))

        return x, w 
    
    def GaussLegendreInterval(self, ng: int, a: np.float64, b: np.float64):
        x0, w0 = roots_jacobi(ng, 0.0, 0.0) 

        xGlobal = np.zeros([ng,1])
        wGlobal = np.zeros([ng,1])

        xGlobal[:,0] = 0.5 * (b - a) * x0 + 0.5 * (b + a)
        wGlobal[:,0] = 0.5 * (b - a) * w0 

        xBdry = np.zeros([2, 1])
        wBdry = np.ones([2, 1])

        xBdry[0,0] = a 
        xBdry[1,0] = b 
        
        self.interior_x = xGlobal 
        self.interior_w = wGlobal 
        self.boundary_x = xBdry 
        self.boundary_w = wBdry 

        return
    
    def GaussLegendreUnitSquare(self, ng: int):
        nx = ng 
        ny = ng

        x0, w0 = roots_jacobi(nx, 0.0, 0.0) 
        w0 = np.reshape(w0, [nx,1])

        x1, w1 = roots_jacobi(ny, 0.0, 0.0) 
        w1 = np.reshape(w1, [ny,1])
    
        xGlobal = np.zeros([nx*ny,2])
        wGlobal = np.zeros([nx*ny,1])

        for i in range(nx):
            for j in range(ny):
                idx = i*(ny) + j
                xGlobal[idx,0] = 0.5 * x0[i] + 0.5 # x in (0,1)
                xGlobal[idx,1] = 0.5 * x1[j] + 0.5 # y in (0,1)
                wGlobal[idx] = (0.5 * w0[i]) * (0.5 * w1[j])

        # boundary quadrature points	
        xBdry = np.zeros([2*nx + 2*ny,2])
        wBdry = np.zeros([2*nx + 2*ny,1])
        tx = np.zeros([2*nx + 2*ny,1])
        ty = np.zeros([2*nx + 2*ny,1])

        # left edge
        xBdry[0:ny,0] = np.zeros([ny,])
        xBdry[0:ny,1] = 0.5 * x1 + 0.5
        wBdry[0:ny,0] = np.squeeze(0.5 * w1)

        # tangent for space
        tx[0:ny,0] = np.zeros([ny,])
        ty[0:ny,0] = np.ones([ny,])

        # right edge
        xBdry[ny:2*ny,0] = np.ones([ny,])
        xBdry[ny:2*ny,1] = 0.5 * x1 + 0.5
        wBdry[ny:2*ny,0] = np.squeeze(0.5 * w1)

        tx[ny:2*ny,0] = np.zeros([ny,])
        ty[ny:2*ny,0] = -np.ones([ny,])

        # bottom edge
        xBdry[2*ny:(2*ny + nx),0] = 0.5 * x0 + 0.5
        xBdry[2*ny:(2*ny + nx),1] = np.zeros([nx,])
        wBdry[2*ny:(2*ny + nx),0] = np.squeeze(0.5 * w0)

        tx[2*ny:(2*ny + nx),0] = -np.ones([nx,])
        ty[2*ny:(2*ny + nx),0] = np.zeros([nx,])

        # top edge
        xBdry[(2*ny + nx):(2*ny + 2*nx),0] = 0.5 * x0 + 0.5
        xBdry[(2*ny + nx):(2*ny + 2*nx),1] = np.ones([nx,])
        wBdry[(2*ny + nx):(2*ny + 2*nx),0] = np.squeeze(0.5 * w0)

        tx[(2*ny + nx):(2*ny + 2*nx),0] = np.ones([nx,])
        ty[(2*ny + nx):(2*ny + 2*nx),0] = np.zeros([nx,])

        self.interior_x = xGlobal 
        self.interior_w = wGlobal 
        self.boundary_x = xBdry 
        self.boundary_w = wBdry 
        self.boundary_t_x = tx 
        self.boundary_t_y = ty

        return
    
    def GaussLegendreRectangle(self, ng: int, a: np.float64, b: np.float64,
                                c: np.float64, d: np.float64):
        Lx = b - a 
        Ly = d - c 

        nx = ng if (Ly > Lx) else int((Lx / Ly) * ng)
        ny = ng if (Lx > Ly) else int((Ly / Lx) * ng)

        x0, w0 = self.GaussLobatto1D(nx) 
        w0 = np.reshape(w0, [nx,1])

        x1, w1 = self.GaussLobatto1D(ny) 
        w1 = np.reshape(w1, [ny,1])
    
        xGlobal = np.zeros([nx*ny,2])
        wGlobal = np.zeros([nx*ny,1])

        for i in range(nx):
            for j in range(ny):
                idx = i*(ny) + j
                xGlobal[idx,0] = 0.5 * (b - a) * x0[i] + 0.5 * (b + a) # x in (a,b)
                xGlobal[idx,1] = 0.5 * (d - c) * x1[j] + 0.5 * (d + c) # y in (c,d)
                wGlobal[idx] = (0.5 * (b - a) * w0[i]) * (0.5 * (d - c) * w1[j])

        # boundary quadrature points	
        xBdry = np.zeros([2*nx + 2*ny,2])
        wBdry = np.zeros([2*nx + 2*ny,1])
        tx = np.zeros([2*nx + 2*ny,1])
        ty = np.zeros([2*nx + 2*ny,1])
        Nx = np.zeros([2*nx + 2*ny,1])
        Ny = np.zeros([2*nx + 2*ny,1])

        # left edge
        xBdry[0:ny,0] = a * np.ones([ny,])
        xBdry[0:ny,1] = 0.5 * (d - c) * x1 + 0.5 * (d + c)
        wBdry[0:ny,0] = np.squeeze(0.5 * (d - c) * w1)

        # tangent and normal for space
        tx[0:ny,0] = np.zeros([ny,])
        ty[0:ny,0] = np.ones([ny,])
        Nx[0:ny,0] = -np.ones([ny,])
        Ny[0:ny,0] = np.zeros([ny,])

        # right edge
        xBdry[ny:2*ny,0] = b * np.ones([ny,])
        xBdry[ny:2*ny,1] = 0.5 * (d - c) * x1 + 0.5 * (d + c)
        wBdry[ny:2*ny,0] = np.squeeze(0.5 * (d - c) * w1)

        tx[ny:2*ny,0] = np.zeros([ny,])
        ty[ny:2*ny,0] = -np.ones([ny,])
        Nx[ny:2*ny,0] = np.ones([ny,])
        Ny[ny:2*ny,0] = np.zeros([ny,])

        # bottom edge
        xBdry[2*ny:(2*ny + nx),0] = 0.5 * (b - a) * x0 + 0.5 * (b + a)
        xBdry[2*ny:(2*ny + nx),1] = c * np.ones([nx,])
        wBdry[2*ny:(2*ny + nx),0] = np.squeeze(0.5 * (b - a) * w0)

        tx[2*ny:(2*ny + nx),0] = -np.ones([nx,])
        ty[2*ny:(2*ny + nx),0] = np.zeros([nx,])
        Nx[2*ny:(2*ny + nx),0] = np.zeros([nx,])
        Ny[2*ny:(2*ny + nx),0] = -np.ones([nx,])

        # top edge
        xBdry[(2*ny + nx):(2*ny + 2*nx),0] = 0.5 * (b - a) * x0 + 0.5 * (b + a)
        xBdry[(2*ny + nx):(2*ny + 2*nx),1] = d * np.ones([nx,])
        wBdry[(2*ny + nx):(2*ny + 2*nx),0] = np.squeeze(0.5 * (b - a) * w0)

        tx[(2*ny + nx):(2*ny + 2*nx),0] = np.ones([nx,])
        ty[(2*ny + nx):(2*ny + 2*nx),0] = np.zeros([nx,])
        Nx[(2*ny + nx):(2*ny + 2*nx),0] = np.zeros([nx,])
        Ny[(2*ny + nx):(2*ny + 2*nx),0] = np.ones([nx,])

        # compatibility condition
        self.compatibility_x = np.zeros([1, 2])
        self.compatibility_w = np.ones([1, 1])

        self.interior_x = xGlobal 
        self.interior_w = wGlobal 
        self.boundary_x = xBdry 
        self.boundary_w = wBdry 
        self.boundary_t_x = tx 
        self.boundary_t_y = ty
        self.boundary_n_x = Nx 
        self.boundary_n_y = Ny

        return 
    
    # quadrature for the annulus with inner radius a, outer radius b
    def GaussLegendreAnnulus(self, ng: int, a: np.float64, b: np.float64):
        nr = ng 
        nt = int(ng * 2.0 * np.pi)

        x0, w0 = roots_jacobi(nr, 0.0, 0.0)
        w0 = np.reshape(w0, [nr,1])

        x1, w1 = roots_jacobi(nt, 0.0, 0.0) 
        w1 = np.reshape(w1, [nt,1])

        xGlobal = np.zeros([nr*nt,2])
        wGlobal = np.zeros([nr*nt,1])

        for i in range(nr):
            for j in range(nt):
                idx = i*(nt) + j

                ri = 0.5 * (b - a) * x0[i] + 0.5 * (b + a) # r in (a,b)
                ti = np.pi * x1[j] + np.pi  # t in (0,2pi)

                x = ri * np.cos(ti)
                y = ri * np.sin(ti)
                xGlobal[idx,0] = x
                xGlobal[idx,1] = y
                wGlobal[idx] = (0.5 * (b - a) * w0[i]) * (np.pi * w1[j])

        # boundary of annulus
        xBdry = np.zeros([3*nt,2])
        wBdry = np.zeros([3*nt,1])
        tx = np.zeros([3*nt,1])
        ty = np.zeros([3*nt,1])
        Nx = np.zeros([3*nt,1])
        Ny = np.zeros([3*nt,1])

        # inner boundary {a} x (0, 2pi)
        t = np.pi * x1 + np.pi
        xBdry[0:nt,0] = a * np.cos(t)
        xBdry[0:nt,1] = a * np.sin(t)
        wBdry[0:nt,0] = np.pi * np.squeeze(w1)

        # tangent and normal for space
        norm = np.sqrt(xBdry[0:nt,0]**2 + xBdry[0:nt,1]**2)
        tx[0:nt,0] = xBdry[0:nt, 1] / norm
        ty[0:nt,0] = -xBdry[0:nt, 0] / norm
        Nx[0:nt,0] = xBdry[0:nt, 0] / norm
        Ny[0:nt,0] = xBdry[0:nt, 1] / norm

        # outer boundary
        x1, w1 = roots_jacobi(2*nt, 0.0, 0.0) 
        w1 = np.reshape(w1, [2*nt,1])

        t = np.pi * x1 + np.pi
        xBdry[nt:3*nt,0] = b * np.cos(t)
        xBdry[nt:3*nt,1] = b * np.sin(t)
        wBdry[nt:3*nt,0] = np.pi * np.squeeze(w1)

        # tangent and normal for space
        norm = np.sqrt(xBdry[nt:3*nt,0]**2 + xBdry[nt:3*nt,1]**2)
        tx[nt:3*nt,0] = xBdry[nt:3*nt, 1] / norm
        ty[nt:3*nt,0] = -xBdry[nt:3*nt, 0] / norm
        Nx[nt:3*nt,0] = xBdry[nt:3*nt, 0] / norm
        Ny[nt:3*nt,0] = xBdry[nt:3*nt, 1] / norm

        self.interior_x = xGlobal 
        self.interior_w = wGlobal 
        self.boundary_x = xBdry 
        self.boundary_w = wBdry 
        self.boundary_t_x = tx 
        self.boundary_t_y = ty
        self.boundary_n_x = Nx 
        self.boundary_n_y = Ny

        return 
    
    # circular sector described by 0 <= r <= 1 and 0 <= t <= theta
    def GaussLegendreCircularSector(self, ng: int, theta: np.float64):
        nr = ng 
        nt = int(ng * theta)

        x0, w0 = roots_jacobi(nr, 0.0, 0.0)
        w0 = np.reshape(w0, [nr,1])

        x1, w1 = roots_jacobi(nt, 0.0, 0.0) 
        w1 = np.reshape(w1, [nt,1])

        xGlobal = np.zeros([nr*nt,2])
        wGlobal = np.zeros([nr*nt,1])

        for i in range(nr):
            for j in range(nt):
                idx = i*(nt) + j

                ri = 0.5 * x0[i] + 0.5  # r in (0,1)
                ti = 0.5 * theta * x1[j] + 0.5 * theta  # t in (0,theta)

                x = ri * np.cos(ti)
                y = ri * np.sin(ti)
                xGlobal[idx,0] = x
                xGlobal[idx,1] = y
                wGlobal[idx] = (0.5 * w0[i]) * (0.5 * theta * w1[j])

        # boundary
        xBdry = np.zeros([2*nr + nt,2])
        wBdry = np.zeros([2*nr + nt,1])
        tx = np.zeros([2*nr + nt,1])
        ty = np.zeros([2*nr + nt,1])
        Nx = np.zeros([2*nr + nt,1])
        Ny = np.zeros([2*nr + nt,1])

        # right edge (0,1) x {0}
        r = 0.5 * x0 + 0.5
        xBdry[0:nr,0] = np.squeeze(r)
        xBdry[0:nr,1] = np.zeros([nr,])
        wBdry[0:nr,0] = np.squeeze(0.5 * w0)

        tx[0:nr,0] = -np.ones([nr,])
        ty[0:nr,0] = np.zeros([nr,])
        Nx[0:nr,0] = np.zeros([nr,])
        Ny[0:nr,0] = -np.ones([nr,])

        # bottom edge (0,1) x {theta}
        xc = np.cos(theta)
        yc = np.sin(theta)
 
        r = 0.5 * x0 - 0.5
        xBdry[nr:2*nr,0] = np.squeeze(0.5 * (0.0-xc) * x0 + 0.5 * (0.0 + xc))
        xBdry[nr:2*nr,1] = np.squeeze(3.0 * (0.5 * (0.0 - xc) * x0 + 0.5 * (0.0 + xc)))
        wBdry[nr:2*nr,0] = np.squeeze(0.5 * (0.0 - xc) * w0)

        tx[nr:2*nr,0] = -np.ones([nr,]) / np.sqrt(10.0)
        ty[nr:2*nr,0] = -3.0*np.ones([nr,]) / np.sqrt(10.0)
        Nx[nr:2*nr,0] = 3.0*np.ones([nr,]) / np.sqrt(10.0)
        Ny[nr:2*nr,0] = -np.ones([nr,]) / np.sqrt(10.0)

        # circular sector segment
        t = 0.5 * theta * x1 + 0.5 * theta
        xBdry[2*nr:2*nr+nt,0] = np.cos(t)
        xBdry[2*nr:2*nr+nt,1] = np.sin(t)
        wBdry[2*nr:2*nr+nt,0] = 0.5 * theta * np.squeeze(w1)
        
        norm = np.sqrt(xBdry[2*nr:2*nr+nt,0]**2 + xBdry[2*nr:2*nr+nt,1]**2)
        tx[2*nr:2*nr+nt,0] = xBdry[2*nr:2*nr+nt, 1] / norm
        ty[2*nr:2*nr+nt,0] = -xBdry[2*nr:2*nr+nt, 0] / norm
        Nx[2*nr:2*nr+nt,0] = xBdry[2*nr:2*nr+nt, 0] / norm
        Ny[2*nr:2*nr+nt,0] = xBdry[2*nr:2*nr+nt, 1] / norm

        self.interior_x = xGlobal 
        self.interior_w = wGlobal 
        self.boundary_x = xBdry 
        self.boundary_w = wBdry 
        self.boundary_t_x = tx 
        self.boundary_t_y = ty
        self.boundary_n_x = Nx 
        self.boundary_n_y = Ny

        self.compatibility_x = np.zeros([1, 2])
        self.compatibility_x[0, 1] = 0.5
        self.compatibility_w = np.ones([1, 1])

        return

    def GaussLegendreLshaped(self, ng: int):
        nx = ng 
        ny = ng

        x0, w0 = roots_jacobi(nx, 0.0, 0.0) 
        # x0, w0 = self.GaussLobatto1D(nx)
        w0 = np.reshape(w0, [nx,1])

        x1, w1 = roots_jacobi(ny, 0.0, 0.0) 
        # x1, w1 = self.GaussLobatto1D(ny)
        w1 = np.reshape(w1, [ny,1])
    
        xGlobal = np.zeros([3*nx*ny,2])
        wGlobal = np.zeros([3*nx*ny,1])

        for i in range(nx):
            for j in range(ny):
                idx = i*(ny) + j
                xGlobal[idx,0] = 0.5 * x0[i] - 0.5 # x in (-1,0)
                xGlobal[idx,1] = 0.5 * x1[j] + 0.5 # y in (0,1)
                wGlobal[idx] = (0.5 * w0[i]) * (0.5 * w1[j])

        for i in range(nx):
            for j in range(ny):
                idx = (nx*ny) + i*ny + j
                xGlobal[idx,0] = 0.5 * x0[i] + 0.5 # x in (0,1)
                xGlobal[idx,1] = 0.5 * x1[j] + 0.5 # y in (0,1)
                wGlobal[idx] = (0.5 * w0[i]) * (0.5 * w1[j])

        for i in range(nx):
            for j in range(ny):
                idx = (2*nx*ny) + i*ny + j
                xGlobal[idx,0] = 0.5 * x0[i] - 0.5 # x in (-1,0)
                xGlobal[idx,1] = 0.5 * x1[j] - 0.5 # y in (-1,0)
                wGlobal[idx] = (0.5 * w0[i]) * (0.5 * w1[j])

        # boundary quadrature points	
        xBdry = np.zeros([4*nx + 4*ny,2])
        wBdry = np.zeros([4*nx + 4*ny,1])
        tx = np.zeros([4*nx + 4*ny,1])
        ty = np.zeros([4*nx + 4*ny,1])

        # top left edge (-1,0) x {1}
        xBdry[0:nx,0] = 0.5 * x0 - 0.5
        xBdry[0:nx,1] = np.ones([nx,])
        wBdry[0:nx,0] = np.squeeze(0.5 * w0)

        # tangent for space
        tx[0:nx,0] = np.ones([nx,])
        ty[0:nx,0] = np.zeros([nx,])

        # top right edge (0,1) x {1}
        xBdry[nx:2*nx,0] = 0.5 * x0 + 0.5
        xBdry[nx:2*nx,1] = np.ones([nx,])
        wBdry[nx:2*nx,0] = np.squeeze(0.5 * w0)

        # tangent for space
        tx[nx:2*nx,0] = np.ones([nx,])
        ty[nx:2*nx,0] = np.zeros([nx,])

        # bottom left edge (-1,0) x {-1}
        xBdry[2*nx:3*nx ,0] = 0.5 * x0 - 0.5
        xBdry[2*nx:3*nx,1] = -np.ones([nx,])
        wBdry[2*nx:3*nx,0] = np.squeeze(0.5 * w0)

        # tangent for space
        tx[2*nx:3*nx,0] = -np.ones([nx,])
        ty[2*nx:3*nx,0] = np.zeros([nx,])

        # middle right edge (0,1) x {0}
        xBdry[3*nx:4*nx ,0] = 0.5 * x0 + 0.5
        xBdry[3*nx:4*nx,1] = np.zeros([nx,])
        wBdry[3*nx:4*nx,0] = np.squeeze(0.5 * w0)

        # tangent for space
        tx[3*nx:4*nx,0] = -np.ones([nx,])
        ty[3*nx:4*nx,0] = np.zeros([nx,])


        # bottom left vertical edge {-1} x (-1,0)
        xBdry[4*nx:(4*nx + ny) ,0] = -np.ones([ny,])
        xBdry[4*nx:(4*nx + ny),1] = 0.5 * x1 - 0.5
        wBdry[4*nx:(4*nx + ny),0] = np.squeeze(0.5 * w1)

        # tangent for space
        tx[4*nx:(4*nx + ny),0] = np.zeros([ny,])
        ty[4*nx:(4*nx + ny),0] = np.ones([ny,])

        # top left vertical edge {-1} x (0,1)
        xBdry[(4*nx + ny):(4*nx + 2*ny),0] = -np.ones([ny,])
        xBdry[(4*nx + ny):(4*nx + 2*ny),1] = 0.5 * x1 + 0.5
        wBdry[(4*nx + ny):(4*nx + 2*ny),0] = np.squeeze(0.5 * w1)

        # tangent for space
        tx[(4*nx + ny):(4*nx + 2*ny),0] = np.zeros([ny,])
        ty[(4*nx + ny):(4*nx + 2*ny),0] = np.ones([ny,])

        # top right vertical edge {1} x (0,1)
        xBdry[(4*nx + 2*ny):(4*nx + 3*ny),0] = np.ones([ny,])
        xBdry[(4*nx + 2*ny):(4*nx + 3*ny),1] = 0.5 * x1 + 0.5
        wBdry[(4*nx + 2*ny):(4*nx + 3*ny),0] = np.squeeze(0.5 * w1)

        # tangent for space
        tx[(4*nx + 2*ny):(4*nx + 3*ny),0] = np.zeros([ny,])
        ty[(4*nx + 2*ny):(4*nx + 3*ny),0] = -np.ones([ny,])

        # middle vertical edge {0} x (-1,0)
        xBdry[(4*nx + 3*ny):(4*nx + 4*ny),0] = np.zeros([ny,])
        xBdry[(4*nx + 3*ny):(4*nx + 4*ny),1] = 0.5 * x1 - 0.5
        wBdry[(4*nx + 3*ny):(4*nx + 4*ny),0] = np.squeeze(0.5 * w1)

        # tangent for space
        tx[(4*nx + 3*ny):(4*nx + 4*ny),0] = np.zeros([ny,])
        ty[(4*nx + 3*ny):(4*nx + 4*ny),0] = -np.ones([ny,])

        self.interior_x = xGlobal 
        self.interior_w = wGlobal 
        self.boundary_x = xBdry 
        self.boundary_w = wBdry 
        self.boundary_t_x = tx 
        self.boundary_t_y = ty

        return
    
    # Gauss-Jacobi quadrature in the L-shaped domain (-1,1)^2 \ (0,1) x (-1,0). 
    # interior is divided into three squares and a non-zero Jacobi parameter is only
    # used in the direction containing the singularity. For example, in (0,1)^2, the
    # singularity is located at (0,0) so a weight function of x^beta for integration in the x-
    # direction and y^beta for intgeration in the y-direction is used.
    #
    # this approach is roughly equivalent to using a weighted least squares formulation
    # with the weight r^beta. if this quadrature rule is used, beta should be set to 0
    # in the GNN class.
    def GaussJacobiLshaped(self, ng: int, beta: np.float64):
        nx = ng 
        ny = ng

        xGlobal = np.zeros([3*nx*ny,2])
        wGlobal = np.zeros([3*nx*ny,1])

        # middle (-1, 0) x (0, 1)
        # interval (-1, 0) gets mapped to (-1, 1) with mapping
        # x |--> 2x + 1, so (1-x)^alpha becomes (-2)^alpha * x^alpha
        # and (1+x)^beta becomes 2^beta * (1+x)^beta
        x0, w0 = roots_jacobi(nx, beta, 0.0) 
        w0 = np.reshape(w0, [nx,1])

        # interval (0, 1) gets mapped to (-1, 1) with mapping
        # x |--> 2x - 1, so (1-x)^alpha becomes 2^alpha * (1-x)^alpha
        # and (1+x)^beta becomes 2^beta * x^beta
        x1, w1 = roots_jacobi(ny, 0.0, beta) 
        w1 = np.reshape(w1, [ny,1])
    
        for i in range(nx):
            for j in range(ny):
                idx = i*(ny) + j
                xGlobal[idx,0] = 0.5 * x0[i] - 0.5 # x in (-1,0)
                xGlobal[idx,1] = 0.5 * x1[j] + 0.5 # y in (0,1)
                wGlobal[idx] = (0.5 * w0[i]) * (0.5 * w1[j])

        # upper right (0, 1)^2
        x0, w0 = roots_jacobi(nx, 0.0, beta) 
        w0 = np.reshape(w0, [nx,1])

        x1, w1 = roots_jacobi(ny, 0.0, beta) 
        w1 = np.reshape(w1, [ny,1])

        for i in range(nx):
            for j in range(ny):
                idx = (nx*ny) + i*ny + j
                xGlobal[idx,0] = 0.5 * x0[i] + 0.5 # x in (0,1)
                xGlobal[idx,1] = 0.5 * x1[j] + 0.5 # y in (0,1)
                wGlobal[idx] = (0.5 * w0[i]) * (0.5 * w1[j])

        # bottom left (-1, 0)^2
        x0, w0 = roots_jacobi(nx, beta, 0.0) 
        w0 = np.reshape(w0, [nx,1])

        x1, w1 = roots_jacobi(ny, beta, 0.0) 
        w1 = np.reshape(w1, [ny,1])

        for i in range(nx):
            for j in range(ny):
                idx = (2*nx*ny) + i*ny + j
                xGlobal[idx,0] = 0.5 * x0[i] - 0.5 # x in (-1,0)
                xGlobal[idx,1] = 0.5 * x1[j] - 0.5 # y in (-1,0)
                wGlobal[idx] = (0.5 * w0[i]) * (0.5 * w1[j])

        # boundary quadrature points	
        xBdry = np.zeros([4*nx + 4*ny,2])
        wBdry = np.zeros([4*nx + 4*ny,1])
        tx = np.zeros([4*nx + 4*ny,1])
        ty = np.zeros([4*nx + 4*ny,1])

        # top left edge (-1,0) x {1}
        x0, w0 = roots_jacobi(nx, 0.0, 0.0) 
        w0 = np.reshape(w0, [nx,1])

        xBdry[0:nx,0] = 0.5 * x0 - 0.5
        xBdry[0:nx,1] = np.ones([nx,])
        wBdry[0:nx,0] = np.squeeze(0.5 * w0)

        # tangent for space
        tx[0:nx,0] = np.ones([nx,])
        ty[0:nx,0] = np.zeros([nx,])

        # top right edge (0,1) x {1}
        xBdry[nx:2*nx,0] = 0.5 * x0 + 0.5
        xBdry[nx:2*nx,1] = np.ones([nx,])
        wBdry[nx:2*nx,0] = np.squeeze(0.5 * w0)

        # tangent for space
        tx[nx:2*nx,0] = np.ones([nx,])
        ty[nx:2*nx,0] = np.zeros([nx,])

        # bottom left edge (-1,0) x {-1}
        xBdry[2*nx:3*nx ,0] = 0.5 * x0 - 0.5
        xBdry[2*nx:3*nx,1] = -np.ones([nx,])
        wBdry[2*nx:3*nx,0] = np.squeeze(0.5 * w0)

        # tangent for space
        tx[2*nx:3*nx,0] = -np.ones([nx,])
        ty[2*nx:3*nx,0] = np.zeros([nx,])

        # middle right edge (0,1) x {0}
        x0, w0 = roots_jacobi(nx, 0.0, beta) 
        w0 = np.reshape(w0, [nx,1])

        xBdry[3*nx:4*nx ,0] = 0.5 * x0 + 0.5
        xBdry[3*nx:4*nx,1] = np.zeros([nx,])
        wBdry[3*nx:4*nx,0] = np.squeeze(0.5 * w0)

        # tangent for space
        tx[3*nx:4*nx,0] = -np.ones([nx,])
        ty[3*nx:4*nx,0] = np.zeros([nx,])


        # bottom left vertical edge {-1} x (-1,0)
        x1, w1 = roots_jacobi(ny, 0.0, 0.0) 
        w1 = np.reshape(w1, [ny,1])

        xBdry[4*nx:(4*nx + ny) ,0] = -np.ones([ny,])
        xBdry[4*nx:(4*nx + ny),1] = 0.5 * x1 - 0.5
        wBdry[4*nx:(4*nx + ny),0] = np.squeeze(0.5 * w1)

        # tangent for space
        tx[4*nx:(4*nx + ny),0] = np.zeros([ny,])
        ty[4*nx:(4*nx + ny),0] = np.ones([ny,])

        # top left vertical edge {-1} x (0,1)
        xBdry[(4*nx + ny):(4*nx + 2*ny),0] = -np.ones([ny,])
        xBdry[(4*nx + ny):(4*nx + 2*ny),1] = 0.5 * x1 + 0.5
        wBdry[(4*nx + ny):(4*nx + 2*ny),0] = np.squeeze(0.5 * w1)

        # tangent for space
        tx[(4*nx + ny):(4*nx + 2*ny),0] = np.zeros([ny,])
        ty[(4*nx + ny):(4*nx + 2*ny),0] = np.ones([ny,])

        # top right vertical edge {1} x (0,1)
        xBdry[(4*nx + 2*ny):(4*nx + 3*ny),0] = np.ones([ny,])
        xBdry[(4*nx + 2*ny):(4*nx + 3*ny),1] = 0.5 * x1 + 0.5
        wBdry[(4*nx + 2*ny):(4*nx + 3*ny),0] = np.squeeze(0.5 * w1)

        # tangent for space
        tx[(4*nx + 2*ny):(4*nx + 3*ny),0] = np.zeros([ny,])
        ty[(4*nx + 2*ny):(4*nx + 3*ny),0] = -np.ones([ny,])

        # middle vertical edge {0} x (-1,0)
        x1, w1 = roots_jacobi(ny, beta, 0.0) 
        w1 = np.reshape(w1, [ny,1])

        xBdry[(4*nx + 3*ny):(4*nx + 4*ny),0] = np.zeros([ny,])
        xBdry[(4*nx + 3*ny):(4*nx + 4*ny),1] = 0.5 * x1 - 0.5
        wBdry[(4*nx + 3*ny):(4*nx + 4*ny),0] = np.squeeze(0.5 * w1)

        # tangent for space
        tx[(4*nx + 3*ny):(4*nx + 4*ny),0] = np.zeros([ny,])
        ty[(4*nx + 3*ny):(4*nx + 4*ny),0] = -np.ones([ny,])

        self.interior_x = xGlobal 
        self.interior_w = wGlobal 
        self.boundary_x = xBdry 
        self.boundary_w = wBdry 
        self.boundary_t_x = tx 
        self.boundary_t_y = ty

        return
    
    # T-shaped channel conisting of channel (-2,2) x (0,2) and triangular cavity
    # with vertices (-1,0), (1,0), and (0,-3).
    def GaussLegendreTshaped(self, ng):

        # rules for channel
        nx = ng 
        ny = ng 
        x0, w0 = roots_jacobi(nx, 0.0, 0.0) 
        w0 = np.reshape(w0, [nx, 1])
        x1, w1 = roots_jacobi(ny, 0.0, 0.0) 
        w1 = np.reshape(w1, [ny, 1])

        # rules for cavity
        nxc = nx
        nyc = int(3*ny/2)
        x0c, w0c = roots_jacobi(nxc, 0.0, 0.0) 
        w0c = np.reshape(w0c, [nxc, 1])
        x1c, w1c = roots_jacobi(nyc, 0.0, 0.0) 
        w1c = np.reshape(w1c, [nyc, 1])

        # rules for long side of cavity
        nside = int(np.sqrt(10.0) / 2.0) * ny
        xside, wside = roots_jacobi(nside, 0.0, 0.0) 
        wside = np.reshape(wside, [nside, 1])
    
        xGlobal = np.zeros([2*nx*ny + nxc*nyc, 2])
        wGlobal = np.zeros([2*nx*ny + nxc*nyc, 1])

        # left quadrant (-2, 0) x (0, 2)
        for i in range(nx):
            for j in range(ny):
                idx = i*ny + j
                xGlobal[idx,0] = x0[i] - 1.0
                xGlobal[idx,1] = x1[j] + 1.0
                wGlobal[idx] = (w0[i]) * (w1[j])

        # upper left quadrant (0, 2) x (0, 2)
        for i in range(nx):
            for j in range(ny):
                idx = (nx*ny) + i*ny + j
                xGlobal[idx, 0] = x0[i] + 1.0
                xGlobal[idx, 1] = x1[j] + 1.0
                wGlobal[idx] = (w0[i]) * (w1[j])

        # bottom quadrant triangle (-1, 1) x (-3x-3, 3x-3)
        for i in range(nxc):
            for j in range(nyc):
                idx = (2*nx*ny) + i*nyc + j
                xGlobal[idx, 0] = x0c[i]
                xGlobal[idx, 1] = 3.0/2.0 * x1c[j] - 3.0/2.0
                wGlobal[idx] = (w0c[i]) * (3.0/2.0 * w1c[j])

        # delete nodes that lie outside of the triangle
        delList = []
        for idx in range(2*nx*ny + nxc*nyc):
            if (xGlobal[idx,1] < 3.0 * xGlobal[idx,0] - 3.0) and (xGlobal[idx,1] < 0.0):
                wGlobal[idx,0] = 0.0
                delList.append(idx)
            elif (xGlobal[idx,1] < -3.0 * xGlobal[idx,0] - 3.0) and (xGlobal[idx,1] < 0.0):
                wGlobal[idx,0] = 0.0
                delList.append(idx)
        wGlobal = np.delete(wGlobal, delList, axis=0)
        xGlobal = np.delete(xGlobal, delList, axis=0)

        # boundary quadrature points
        xBdry = np.zeros([4*nx + 2*ny + 2*nside, 2])
        wBdry = np.zeros([4*nx + 2*ny + 2*nside, 1])
        tx = np.zeros([4*nx + 2*ny + 2*nside, 1])
        ty = np.zeros([4*nx + 2*ny + 2*nside, 1])
        Nx = np.zeros([4*nx + 2*ny + 2*nside, 1])
        Ny = np.zeros([4*nx + 2*ny + 2*nside, 1])

        # top left edge (-2, 0) x {2}
        xBdry[0:nx,0] = np.squeeze(2.0*x0 - 1.0)
        xBdry[0:nx,1] = 2.0*np.ones([nx,])
        wBdry[0:nx] = 2.0*w0
        tx[0:nx,0] = np.ones([nx,])
        ty[0:nx,0] = np.zeros([nx,])
        Nx[0:nx,0] = np.zeros([nx,])
        Ny[0:nx,0] = np.ones([nx,])

        # top right edge (0, 2) x {2}
        xBdry[nx:2*nx,0] = np.squeeze(2.0*x0 + 1.0)
        xBdry[nx:2*nx,1] = 2.0*np.ones([nx,])
        wBdry[nx:2*nx] = 2.0*w0
        tx[nx:2*nx,0] = np.ones([nx,])
        ty[nx:2*nx,0] = np.zeros([nx,])
        Nx[nx:2*nx,0] = np.zeros([nx,])
        Ny[nx:2*nx,0] = np.ones([nx,])

        # middle left edge (-2,-1) x {0}
        xBdry[2*nx:3*nx,0] = np.squeeze(0.5*x0 - 1.5)
        xBdry[2*nx:3*nx,1] = np.zeros([nx,])
        wBdry[2*nx:3*nx] = 0.5*w0
        tx[2*nx:3*nx,0] = -np.ones([nx,])
        ty[2*nx:3*nx,0] = np.zeros([nx,])
        Nx[2*nx:3*nx,0] = np.zeros([nx,])
        Ny[2*nx:3*nx,0] = -np.ones([nx,])

        # middle right edge (1, 2) x {0}
        xBdry[3*nx:4*nx,0] = np.squeeze(0.5*x0 + 1.5)
        xBdry[3*nx:4*nx,1] = np.zeros([nx,])
        wBdry[3*nx:4*nx] = 0.5*w0
        tx[3*nx:4*nx,0] = -np.ones([nx,])
        ty[3*nx:4*nx,0] = np.zeros([nx,])
        Nx[3*nx:4*nx,0] = np.zeros([nx,])
        Ny[3*nx:4*nx,0] = -np.ones([nx,])

        # right edge {2} x (0, 2)
        xBdry[4*nx:(4*nx + ny),0] = 2.0 * np.ones([ny,])
        xBdry[4*nx:(4*nx + ny),1] = np.squeeze(x1 + 1.0)
        wBdry[4*nx:(4*nx + ny)] = w1
        tx[4*nx:(4*nx + ny),0] = np.zeros([nx,])
        ty[4*nx:(4*nx + ny),0] = -np.ones([nx,])
        Nx[4*nx:(4*nx + ny),0] = np.ones([nx,])
        Ny[4*nx:(4*nx + ny),0] = np.zeros([nx,])

        # left edge {-2} x (0, 2)
        xBdry[(4*nx + ny):(4*nx + 2*ny),0] = -2.0 * np.ones([ny,])
        xBdry[(4*nx + ny):(4*nx + 2*ny),1] = np.squeeze(x1 + 1.0)
        wBdry[(4*nx + ny):(4*nx + 2*ny)] = w1
        tx[(4*nx + ny):(4*nx + 2*ny),0] = np.zeros([nx,])
        ty[(4*nx + ny):(4*nx + 2*ny),0] = np.ones([nx,])
        Nx[(4*nx + ny):(4*nx + 2*ny),0] = -np.ones([nx,])
        Ny[(4*nx + ny):(4*nx + 2*ny),0] = np.zeros([nx,])

        # right cavity edge {3x-3} x (-3, 0)
        xBdry[(4*nx + 2*ny):(4*nx + 2*ny + nside),0] = np.squeeze(0.5*xside + 0.5)
        xBdry[(4*nx + 2*ny):(4*nx + 2*ny + nside),1] = np.squeeze(3.0*(0.5*xside + 0.5) - 3.0)
        wBdry[(4*nx + 2*ny):(4*nx + 2*ny + nside)] = 0.5*wside
        tx[(4*nx + 2*ny):(4*nx + 2*ny + nside),0] = -np.ones([nside,]) / np.sqrt(10.0)
        ty[(4*nx + 2*ny):(4*nx + 2*ny + nside),0] = -3.0*np.ones([nside,]) / np.sqrt(10.0)
        Nx[(4*nx + 2*ny):(4*nx + 2*ny + nside),0] = 3.0*np.ones([nside,]) / np.sqrt(10.0)
        Ny[(4*nx + 2*ny):(4*nx + 2*ny + nside),0] = -np.ones([nside,]) / np.sqrt(10.0)

        # left cavity edge
        xBdry[(4*nx + 2*ny + nside):(4*nx + 2*ny + 2*nside),0] = np.squeeze(0.5*xside - 0.5)
        xBdry[(4*nx + 2*ny + nside):(4*nx + 2*ny + 2*nside),1] = np.squeeze(-3.0*(0.5*xside - 0.5) - 3.0)
        wBdry[(4*nx + 2*ny + nside):(4*nx + 2*ny + 2*nside)] = 0.5*wside
        tx[(4*nx + 2*ny + nside):(4*nx + 2*ny + 2*nside),0] = -np.ones([nside,]) / np.sqrt(10.0)
        ty[(4*nx + 2*ny + nside):(4*nx + 2*ny + 2*nside),0] = 3.0*np.ones([nside,]) / np.sqrt(10.0)
        Nx[(4*nx + 2*ny + nside):(4*nx + 2*ny + 2*nside),0] = -3.0*np.ones([nside,]) / np.sqrt(10.0)
        Ny[(4*nx + 2*ny + nside):(4*nx + 2*ny + 2*nside),0] = -np.ones([nside,]) / np.sqrt(10.0)

        self.interior_x = xGlobal 
        self.interior_w = wGlobal 
        self.boundary_x = xBdry 
        self.boundary_w = wBdry 
        self.boundary_t_x = tx 
        self.boundary_t_y = ty
        self.boundary_n_x = Nx 
        self.boundary_n_y = Ny
        self.compatibility_x = np.zeros([1, 2])
        self.compatibility_x[0, 1] = 1.0
        self.compatibility_w = np.ones([1, 1])

        return
    
    # triangular wedge with vertices (-1,0), (1,0), and (0,-3)
    def GaussLegendreTriangularWedge(self, ng: int):
        nx = ng 
        ny = ng 

        # rules for cavity
        nxc = nx
        nyc = int(3*ny/2)
        x0c, w0c = roots_jacobi(nxc, 0.0, 0.0) 
        w0c = np.reshape(w0c, [nxc, 1])
        x1c, w1c = roots_jacobi(nyc, 0.0, 0.0) 
        w1c = np.reshape(w1c, [nyc, 1])

        # rules for long side of cavity
        nside = int(np.sqrt(10.0) / 2.0) * ny
        xside, wside = roots_jacobi(nside, 0.0, 0.0) 
        wside = np.reshape(wside, [nside, 1])

        xGlobal = np.zeros([nxc*nyc, 2])
        wGlobal = np.zeros([nxc*nyc, 1])

        # bottom quadrant triangle (-1, 1) x (-3x-3, 3x-3)
        for i in range(nxc):
            for j in range(nyc):
                idx = i*nyc + j
                xGlobal[idx, 0] = x0c[i]
                xGlobal[idx, 1] = 3.0/2.0 * x1c[j] - 3.0/2.0
                wGlobal[idx] = (w0c[i]) * (3.0/2.0 * w1c[j])

        # delete nodes that lie outside of the triangle
        delList = []
        for idx in range(nxc*nyc):
            if (xGlobal[idx,1] < 3.0 * xGlobal[idx,0] - 3.0) and (xGlobal[idx,1] < 0.0):
                wGlobal[idx,0] = 0.0
                delList.append(idx)
            elif (xGlobal[idx,1] < -3.0 * xGlobal[idx,0] - 3.0) and (xGlobal[idx,1] < 0.0):
                wGlobal[idx,0] = 0.0
                delList.append(idx)
        wGlobal = np.delete(wGlobal, delList, axis=0)
        xGlobal = np.delete(xGlobal, delList, axis=0)

        # boundary quadrature points
        xBdry = np.zeros([nxc + 2*nside, 2])
        wBdry = np.zeros([nxc + 2*nside, 1])
        tx = np.zeros([nxc + 2*nside, 1])
        ty = np.zeros([nxc + 2*nside, 1])
        Nx = np.zeros([nxc + 2*nside, 1])
        Ny = np.zeros([nxc + 2*nside, 1])

        # top edge
        xBdry[0:nxc,0] = np.squeeze(x0c)
        xBdry[0:nxc,1] = np.zeros([nxc,])
        wBdry[0:nxc,0] = np.squeeze(w0c)
        tx[0:nxc,0] = np.ones([nxc,])
        ty[0:nxc,0] = np.zeros([nxc,])
        Nx[0:nxc,0] = np.zeros([nxc,])
        Ny[0:nxc,0] = np.ones([nxc,])

        # right cavity edge {3x-3} x (-3, 0)
        xBdry[nxc:(nxc + nside),0] = np.squeeze(0.5*xside + 0.5)
        xBdry[nxc:(nxc + nside),1] = np.squeeze(3.0*(0.5*xside + 0.5) - 3.0)
        wBdry[nxc:(nxc + nside)] = 0.5*wside
        tx[nxc:(nxc + nside),0] = -np.ones([nside,]) / np.sqrt(10.0)
        ty[nxc:(nxc + nside),0] = -3.0*np.ones([nside,]) / np.sqrt(10.0)
        Nx[nxc:(nxc + nside),0] = 3.0*np.ones([nside,]) / np.sqrt(10.0)
        Ny[nxc:(nxc + nside),0] = -np.ones([nside,]) / np.sqrt(10.0)

        # left cavity edge
        xBdry[(nxc + nside):(nxc + 2*nside),0] = np.squeeze(0.5*xside - 0.5)
        xBdry[(nxc + nside):(nxc + 2*nside),1] = np.squeeze(-3.0*(0.5*xside - 0.5) - 3.0)
        wBdry[(nxc + nside):(nxc + 2*nside)] = 0.5*wside
        tx[(nxc + nside):(nxc + 2*nside),0] = -np.ones([nside,]) / np.sqrt(10.0)
        ty[(nxc + nside):(nxc + 2*nside),0] = 3.0*np.ones([nside,]) / np.sqrt(10.0)
        Nx[(nxc + nside):(nxc + 2*nside),0] = -3.0*np.ones([nside,]) / np.sqrt(10.0)
        Ny[(nxc + nside):(nxc + 2*nside),0] = -np.ones([nside,]) / np.sqrt(10.0)

        self.interior_x = xGlobal 
        self.interior_w = wGlobal 
        self.boundary_x = xBdry 
        self.boundary_w = wBdry 
        self.boundary_t_x = tx 
        self.boundary_t_y = ty
        self.boundary_n_x = Nx 
        self.boundary_n_y = Ny
        self.compatibility_x = np.zeros([1, 2])
        self.compatibility_x[0, 1] = -1.0
        self.compatibility_w = np.ones([1, 1])

        return
    
    def GaussLegendreSpaceTimeRectangle(self, ng: int, ax: np.float64, bx: np.float64,
                                        at: np.float64, bt: np.float64):
        Lx = bx - ax 
        Lt = bt - at 

        if np.minimum(Lx, Lt) == Lx:
            nx = ng 
            nt = int(ng * Lt / Lx)
        else:
            nt = ng 
            nx = int(ng * Lx / Lt)

        x0, w0 = roots_jacobi(nx, 0.0, 0.0) 
        w0 = np.reshape(w0, [nx,1])

        x1, w1 = roots_jacobi(nt, 0.0, 0.0) 
        w1 = np.reshape(w1, [nt,1])

        xGlobal = np.zeros([nx*nt,2])
        wGlobal = np.zeros([nx*nt,1])

        for i in range(nx):
            for j in range(nt):
                idx = i*(nt) + j
                xGlobal[idx,0] = np.pi * x0[i] + np.pi # x in (0,2pi)
                xGlobal[idx,1] = 0.5 * x1[j] + 0.5 	   # t in (0,1)
                wGlobal[idx] = (np.pi * w0[i]) * (0.5 * w1[j])

        # boundary quadrature points	
        xBdry = np.zeros([nt,2])
        wBdry = np.zeros([nt,1])
        tx = np.zeros([nt,1])
        ty = np.zeros([nt,1])
        tx_t = np.zeros([nt,1])
        ty_t = np.zeros([nt,1])

        # left face
        xBdry[0:nt,0] = np.zeros([nt,])
        xBdry[0:nt,1] = 0.5 * x1 + 0.5
        wBdry[0:nt,0] = np.squeeze(0.5 * w1)

        # tangent for space
        tx[0:nt,0] = np.ones([nt,])
        ty[0:nt,0] = np.zeros([nt,])

        # tangent for time
        tx_t[0:nt,0] = np.zeros([nt,])
        ty_t[0:nt,0] = np.ones([nt,])


        # initial condition at t=0
        tBdry = np.zeros([nx,2])
        wtBdry = np.zeros([nx,1])

        x0, w0 = roots_jacobi(nx, 0.0, 0.0) 
        w0 = np.reshape(w0, [nx,1])

        # bottom edge (0,2pi) x {0}
        tBdry[:,0] = np.pi * x0 + np.pi
        tBdry[:,1] = np.zeros([nx,])
        wtBdry[:,0] = np.squeeze(np.pi * w0)

        self.interior_w = wGlobal 
        self.interior_x = xGlobal 
        self.boundary_w = wBdry 
        self.boundary_x = xBdry 
        self.boundary_t_x = tx 
        self.boundary_t_t = ty 
        self.initial_x = tBdry 
        self.initial_w = wtBdry
        self.initial_t_x = tx_t 
        self.initial_t_t = ty_t 
         
        return 