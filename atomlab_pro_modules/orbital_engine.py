
# orbital_engine.py
import numpy as np
from scipy.special import sph_harm, genlaguerre

# Hydrogenic radial wavefunction (unnormalized) R_{n,l}(r) in atomic units (a0=1)
# For visualization we'll compute probability density |psi|^2

def radial_part(n, l, r):
    # n: principal, l: orbital
    rho = 2.0 * r / n
    # Associated Laguerre L_{n-l-1}^{2l+1}(rho)
    k = n - l - 1
    if k < 0:
        return np.zeros_like(r)
    L = genlaguerre(k, 2*l+1)(rho)
    prefactor = (2.0/n)**3 * math.factorial(k) / (2*n*math.factorial(n + l)) if (n+l)>=0 else (2.0/n)**3
    R = (rho**l) * np.exp(-rho/2.0) * L
    return R

def hydrogen_wavefunction(n, l, m, r, theta, phi):
    # radial * angular
    R = radial_part(n,l,r)
    Y = sph_harm(m, l, phi, theta)
    psi = R * Y
    return psi

