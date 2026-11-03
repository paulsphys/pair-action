import numpy as np
import scipy.special as scp
from scipy import integrate

#Aziz Potential and it's derivative
def F(x,D):
    a = np.zeros(np.shape(x))
    it = np.nditer(x, flags=['multi_index'])
    for i in it:
        if (i < D):
            a[it.multi_index]  = np.exp(-(D/i - 1.0)*(D/i - 1.0))
        else:
            a[it.multi_index] = 1.0
    return a
def dF(x,D):
    ix = 1.0 / x
    r = 2.0 * D * ix**2 * (D * ix - 1.0) * np.exp(-(D * ix - 1.0)**2)
    return np.where(x < D, r, 0.0)
def Aziz_potential(r):
    EPS     = 1.0E-7
    rm      = 2.9683
    A       = 0.5448504E6;
    epsilon = 10.8;
    alpha   = 13.353384
    D       = 1.241314;
    C6      = 1.3732412;
    C8      = 0.4253785;
    C10     = 0.1781;

    x = r / rm

    Urep = A * np.exp(-alpha*x);
    if (x < EPS):
        return 0.0;
    elif (x < 0.01):
        return (epsilon * Urep);
    else:
        ix2 = 1.0 / (x * x);
        ix6 = ix2 * ix2 * ix2;
        ix8 = ix6 * ix2;
        ix10 = ix8 * ix2;
        Uatt = -( C6*ix6 + C8*ix8 + C10*ix10 ) * F(x,D);
        val = epsilon * (Urep + Uatt)
    return float(val)
def free_prop_1d(x1,x2, tau, lam):
    return (1.0 / np.sqrt(4.0 * np.pi * lam * tau)) * np.exp(-(x1 - x2) ** 2 / (4.0 * lam * tau))

_vbar_cache = {}
def avg_V_matrix(r):
    #line-averaged potential on the whole grid,
    key = (r[0], r[-1], len(r))
    if key not in _vbar_cache:
        N = len(r)
        Vb = np.zeros((N, N))
        for a in range(N):
            Vb[a, a] = Aziz_potential(r[a])
            for b in range(a+1, N):
                Vb[a, b] = Vb[b, a] = integrate.quad(Aziz_potential, r[a], r[b])[0] / (r[b] - r[a])
        _vbar_cache[key] = Vb
    return _vbar_cache[key]

def free_part(l,r,rpr,tau,lam,dim):
    if(dim == 1):
        rhofree = free_prop_1d(r,rpr,tau,lam)
        dlt = -1/(2*tau) + (r - rpr)**2/(4*lam*tau**2)
        dll = -1/(2*lam) + (r - rpr)**2/(4*lam**2*tau)
        return rhofree, dlt, dll
    arg = r*rpr/(2*lam*tau)
    expfac = np.exp(-(r - rpr)**2/(4*lam*tau))
    if(dim == 2):
        coeff = 2*np.pi*np.sqrt(r*rpr) / (4*np.pi*lam*tau)
        bes = scp.ive(l,arg)
        dlogbes = 0.5*(scp.ive(l-1,arg) + scp.ive(l+1,arg))/bes - 1          # d ln(ive)/d arg
    else:
        coeff = 4 * np.pi * r*rpr / np.power(4*np.pi*lam*tau,3/2)
        nu = l + 0.5                                                          # i_l(x) = sqrt(pi/2x) I_{l+1/2}(x)
        bes = np.sqrt(np.pi/(2*arg)) * scp.ive(nu,arg)                        # = i_l(x) e^{-x}
        dlogbes = -1/(2*arg) + 0.5*(scp.ive(nu-1,arg) + scp.ive(nu+1,arg))/scp.ive(nu,arg) - 1
    rhofree = coeff * expfac * bes
    dlt = -dim/(2*tau) + ((r - rpr)**2/(4*lam*tau**2)) - arg*dlogbes/tau
    dll = -dim/(2*lam) + ((r - rpr)**2/(4*lam**2*tau)) - arg*dlogbes/lam
    return rhofree, dlt, dll

def highT_action(l,r,rpr,tau,lam,dim,avg_V):
    """
    Function which returns the high T approximation for the density matrix
    """
    rhofree, _, _ = free_part(l,r,rpr,tau,lam,dim)
    return rhofree * np.exp(-tau*avg_V)

def highT_action_dtau(l,r,rpr,tau,lam,dim,avg_V):
    """
    Function which returns the tau derivative high T approximation for the density matrix
    """
    rhofree, dlt, _ = free_part(l,r,rpr,tau,lam,dim)
    return rhofree * np.exp(-tau*avg_V) * (dlt - avg_V)         

def highT_action_dlam(l,r,rpr,tau,lam,dim,avg_V):
    """
    Function which returns the lambda derivative high T approximation for the density matrix
    """
    rhofree, _, dll = free_part(l,r,rpr,tau,lam,dim)
    return rhofree * np.exp(-tau*avg_V) * dll                    

#Build low tau action by squaring
def build_pair_table(l, tau, n_d, r, lam, dim, _pair_table_cache):
    key = (l, round(tau, 12), n_d, r[0], r[-1], len(r))
    if key in _pair_table_cache:
        return _pair_table_cache[key]
    dr = r[1] - r[0]
    taui = tau / 2**n_d
    R, Rp = r[:, None], r[None, :]                                
    Vb = avg_V_matrix(r)
    rho  = highT_action(l, R, Rp, taui, lam, dim, Vb)
    dt   = highT_action_dtau(l, R, Rp, taui, lam, dim, Vb)
    dlam = highT_action_dlam(l, R, Rp, taui, lam, dim, Vb)
    for _ in range(n_d):
        rho, dt, dlam = (rho @ rho)*dr, 0.5*(dt @ rho + rho @ dt)*dr, (dlam @ rho + rho @ dlam)*dr
    _pair_table_cache[key] = (rho, dt, dlam)                      
    return rho,dt,dlam

#Do a partial wave summation
def partial_wave_summation(rvec,rprvec,grid,lam,tau,n_d,order,dim,cache):
    a, b = np.linalg.norm(rvec), np.linalg.norm(rprvec)
    i, j = np.argmin(np.abs(grid - a)), np.argmin(np.abs(grid - b))
    if(dim == 1):                                                 
        return tuple(t[i, j] for t in build_pair_table(0, tau, n_d, grid, lam, dim, cache))
    theta = np.arccos(np.clip(np.dot(rvec,rprvec)/(a*b), -1, 1))
    val = 0
    valtau = 0
    vallam = 0
    if(dim == 2):
        coeff = 1/(2*np.pi*np.sqrt(a*b))
        for l in range(-order,order+1):
            values = [t[i, j] for t in build_pair_table(abs(l), tau, n_d, grid, lam, dim, cache)]
            val += values[0]*np.exp(1j*l*theta)
            valtau += values[1]*np.exp(1j*l*theta)
            vallam += values[2]*np.exp(1j*l*theta)
    elif(dim == 3):
        coeff = 1/(4*np.pi*a*b)
        for l in range(0,order+1):                                
            values = [t[i, j] for t in build_pair_table(l, tau, n_d, grid, lam, dim, cache)]
            P = (2*l+1)*scp.eval_legendre(l, np.cos(theta))
            val += values[0]*P
            valtau += values[1]*P
            vallam += values[2]*P
    return (coeff*val).real,(coeff*valtau).real,(coeff*vallam).real

def ln_rho0(rvec,rprvec,lam,tau,dim):
    s2 = np.sum((rvec-rprvec)**2)
    return -s2/(4*lam*tau) - 0.5*dim*np.log(4*np.pi*lam*tau), s2

#Pair action u = -ln(rho/rho_free)
def pair_action(rvec,rprvec,grid,lam,tau,n_d,order,dim,cache):
    rho = partial_wave_summation(rvec,rprvec,grid,lam,tau,n_d,order,dim,cache)[0]
    lr0, _ = ln_rho0(rvec,rprvec,lam,tau,dim)
    return -np.log(rho) + lr0

# du/dtau = -(drho/dtau)/rho + d ln(rho0)/dtau
def pair_action_dtau(rvec,rprvec,grid,lam,tau,n_d,order,dim,cache):
    rho, drho, _ = partial_wave_summation(rvec,rprvec,grid,lam,tau,n_d,order,dim,cache)
    _, s2 = ln_rho0(rvec,rprvec,lam,tau,dim)
    return -drho/rho - dim/(2*tau) + s2/(4*lam*tau**2)

# du/dlam = -(drho/dlam)/rho + d ln(rho0)/dlam
def pair_action_dlam(rvec,rprvec,grid,lam,tau,n_d,order,dim,cache):
    rho, _, drho = partial_wave_summation(rvec,rprvec,grid,lam,tau,n_d,order,dim,cache)
    _, s2 = ln_rho0(rvec,rprvec,lam,tau,dim)
    return -drho/rho - dim/(2*lam) + s2/(4*lam**2*tau)


def basis(s,z,n,dim):
    if(dim == 1):
        return [s**(2*k) for k in range(1,n+1)]
    return [z**(2*j) * s**(2*(k-j)) for k in range(1,n+1) for j in range(k+1)]

def vec(x, th, dim):
    if(dim == 1):
        return np.array([x])
    v = np.zeros(dim)
    v[0], v[1] = x*np.cos(th), x*np.sin(th)
    return v


def fit_ukj(grid,lam,tau,n_d,order,n,func,dim,cache,m_max=5,n_theta=10):
    N = len(grid)
    ud = np.array([func(vec(g,0,dim),vec(g,0,dim),grid,lam,tau,n_d,order,dim,cache) for g in grid])

    width = np.sqrt(2*lam*tau)
    step = max(1, int(round(4*width/(2*m_max*(grid[1]-grid[0])))))
    qs, coeffs, resid = [], [], []
    for iq in range(N):
        thetas = [0.0] if dim == 1 else np.linspace(0, min(np.pi, 4*width/grid[iq]), n_theta)
        A, y, sA = [], [], []
        for m in range(m_max+1):
            ia, ib = iq + m*step, iq - m*step
            if ib < 0 or ia >= N:
                continue
            a, b = grid[ia], grid[ib]
            for th in thetas:
                rvec = vec(a,0,dim)
                rprvec = vec(b,th,dim)
                s = np.linalg.norm(rvec - rprvec)
                z = a - b
                y.append(func(rvec,rprvec,grid,lam,tau,n_d,order,dim,cache) - (ud[ia] + ud[ib])/2)
                A.append(basis(s,z,n,dim))
                sA.append(s*s)
        A, y = np.array(A), np.array(y)
        ok = np.isfinite(y)                                          
 
        w = np.exp(-np.array(sA)/(8*lam*tau))[ok]
        c, *_ = np.linalg.lstsq(A[ok]*w[:, None], y[ok]*w, rcond=None)
        qs.append(grid[iq])
        coeffs.append(c)
        resid.append(np.max(w**2*np.abs(A[ok] @ c - y[ok])) if ok.any() else 0.0)   
    return np.array(qs), np.array(coeffs), ud, np.array(resid)

def eval_U(rvec, rprvec, qs, coeffs, grid, ud, n, dim):
    a, b = np.linalg.norm(rvec), np.linalg.norm(rprvec)
    q = (a + b)/2
    s = np.linalg.norm(rvec - rprvec)
    z = a - b
    bs = basis(s,z,n,dim)                                           
    ukj = [np.interp(q, qs, coeffs[:,i]) for i in range(len(bs))]
    return (np.interp(a, grid, ud) + np.interp(b, grid, ud))/2 + np.dot(ukj, bs)
