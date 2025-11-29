
# bohr_model.py
import math
from math import cos, sin, pi

def wavelength_ev_to_nm(E_eV):
    # h (eV*s) * c (m/s) / E (eV) -> meters -> nm
    h_eV_s = 4.135667696e-15
    c = 299792458.0
    if E_eV <= 0:
        return None
    lam_m = (h_eV_s * c) / E_eV
    return lam_m * 1e9

# Map visible wavelength (380..750 nm) to approximate RGB
def wavelength_to_rgb(wavelength_nm):
    if wavelength_nm is None:
        return (255,255,255)
    w = float(wavelength_nm)
    gamma = 0.8
    R=G=B=0.0
    if 380 <= w <= 440:
        R = -(w - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= w <= 490:
        R = 0.0
        G = (w - 440) / (490 - 440)
        B = 1.0
    elif 490 <= w <= 510:
        R = 0.0
        G = 1.0
        B = -(w - 510) / (510 - 490)
    elif 510 <= w <= 580:
        R = (w - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= w <= 645:
        R = 1.0
        G = -(w - 645) / (645 - 580)
        B = 0.0
    elif 645 <= w <= 750:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R=G=B=0.0
    # Intensity correction
    if 380 <= w <= 420:
        factor = 0.3 + 0.7*(w - 380)/(420-380)
    elif 420 <= w <= 700:
        factor = 1.0
    elif 700 <= w <= 750:
        factor = 0.3 + 0.7*(750 - w)/(750-700)
    else:
        factor = 0.0
    def adjust(c):
        if c==0.0:
            return 0
        return int(round((c*factor)**gamma * 255))
    return (adjust(R), adjust(G), adjust(B))
