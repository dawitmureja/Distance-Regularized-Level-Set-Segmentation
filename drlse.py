import numpy as np
import ipdb
def del2(M):
    r,c = M.shape
    D = np.zeros(M.shape)
    for i in range(r):
        for j in range(c):
            if (i == 0 and j == 0):
                D[i,j] = -5 * M[i,j+1] + 4*M[i,j + 2] - M[i,j+3] - 5*M[i+1,j] + 4*M[i+2,j]-M[i+3,j] + 4 *M[i,j]
            elif (i == 0 and j == c-1):
                D[i,j] = -5 * M[i,j-1] + 4*M[i,j - 2] - M[i,j-3] - 5*M[i+1,j] + 4*M[i+2,j]-M[i+3,j] + 4*M[i,j]
            elif(i == r-1 and j == 0):
                D[i,j] = -5 * M[i,j+1] + 4*M[i,j+2] - M[i,j+3] - 5*M[i-1,j] + 4*M[i-2,j]-M[i-3,j] + 4*M[i,j]
            elif (i == r-1 and j == c-1):
                D[i,j] = -5 * M[i,j-1] + 4*M[i,j - 2] - M[i,j-3] - 5*M[i-1,j] + 4*M[i-2,j]-M[i-3,j] + 4*M[i,j]
            elif (i == r-1):
                D[i,j] = -5 * M[i-1,j] + 4*M[i-2,j] - M[i-3,j] + M[i,j+1] + M[i,j-1]
            elif(i == 0):
                D[i,j] = -5 * M[i+1,j] + 4*M[i+2,j] - M[i+3,j] + M[i,j+1] + M[i,j-1]
            elif(j == 0):
                D[i,j] = -5 * M[i,j+1] + 4*M[i,j+2] - M[i,j+3] + M[i+1,j] + M[i-1,j]
            elif (j == c-1):
                D[i,j] = -5 * M[i,j-1] + 4*M[i,j-2] - M[i,j-3] + M[i+1,j] + M[i-1,j]
            else:
                #D[i,j] = M[i+1,j] + M[i-1,j] -2*M[i,j]
                D[i,j] = M[i+1,j] + M[i-1,j]+ M[i,j+1] + M[i, j-1] -4*M[i,j]

    return D/4.0

def gauss2D(shape=(3,3),sigma=0.5):

    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def choose_boundary(a,b,c,d):
    rows = np.array([a,b], dtype = np.int)
    cols = np.array([c,d], dtype = np.int)
    return [rows[:,np.newaxis],cols]

def multiple_choose(a,b,c,d, m_row = False):
    if m_row:
        rows = np.array(range(a,b), dtype = np.int)
        cols = np.array([c,d], dtype = np.int)
        return [rows[:,np.newaxis],cols]
    else:
        rows = np.array([a,b], dtype = np.int)
        cols = np.array(range(c,d), dtype = np.int)
        return [rows[:,np.newaxis],cols]

def NeumannBoundaryCond(f):
    #make a function statisfy Neumann boundary condition
    nrow,ncol = f.shape
    g = f
    g[choose_boundary(0,nrow-1,0,ncol-1)] = g[choose_boundary(2,nrow-3,2,ncol-3)]
    g[multiple_choose(0, nrow-1,1,ncol-1, m_row = False)] = g[multiple_choose(2,nrow-3,1, ncol-1,m_row = False)]
    g[multiple_choose(1, nrow-1,0,ncol-1, m_row = True)] = g[multiple_choose(1,nrow-1,2, ncol-3,m_row = True)]
    return g

def Dirac(x,sigma):
    f = (1/(2 * sigma)) * (1 + np.cos((np.pi * x)/sigma))
    b = np.all([x <= sigma, x >= -sigma], axis = 0)
    return f * b

def div(nx,ny):
    nxx,junk = np.gradient(nx)
    junk,nyy = np.gradient(ny)
    return nxx + nyy

def distReg_p2(phi):
    phi_x,phi_y = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    a = np.all([s >= 0, s <= 1], axis = 0)
    b = s > 1
    ps = a * np.sin(2*np.pi*s)/ (2 * np.pi) + b * (s-1)
    dps = ((ps != 0) * ps + (ps == 0))/((s != 0) * s + (s == 0))
    f = div(dps * phi_x - phi_x, dps * phi_y -phi_y) + 4 * del2(phi)
    return f

def drlse_edge(phi_0,g,lamda,mu,alfa,epsilon, timestep, iteration, potentialFunction):
    phi = phi_0
    vx,vy = np.gradient(g)
    for k in range(iteration):
        phi = NeumannBoundaryCond(phi)
        phi_x,phi_y = np.gradient(phi)
        s = np.sqrt(np.square(phi_x) + np.square(phi_y))
        smallNumber = 1e-10
        Nx = phi_x / (s+smallNumber)
        Ny = phi_y / (s+smallNumber)
        curvature = div(Nx,Ny)
        if (potentialFunction == 'single-well'):
            distRegTerm = 4 * del2(phi) - curvature
        elif(potentialFunction == 'double-well'):
            distRegTerm = distReg_p2(phi)
        else:
            printf('Error: Wrong choice of potential function')
        diracPhi = Dirac(phi,epsilon)
        areaTerm= diracPhi * g
        edgeTerm =  diracPhi * (vx * Nx + vy * Ny) + diracPhi * g * curvature
        phi += timestep * (mu * distRegTerm + lamda * edgeTerm + alfa * areaTerm)
    return phi


#print(np.gradient(np.arange(12).reshape(3,4)))
#print(Dirac(np.array([.1,.2,.3,.4]),0.5))
#a = np.arange(25).reshape(5,5)
#print(a)
#b = NeumannBoundaryCond(a)
#print(b)
