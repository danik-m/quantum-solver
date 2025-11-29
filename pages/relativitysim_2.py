# barrier_only.py
# ТІЛЬКИ БАР'ЄР — строго без жодних змін (як у твоєму оригіналі)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants, linalg

# -------------------------------------------------------------------------
# КОНСТАНТИ
# -------------------------------------------------------------------------
HBAR = constants.hbar
M_E = constants.m_e
EV = constants.electron_volt

# -------------------------------------------------------------------------
# БАР'ЄР — ТОЧНО ЯК БУЛО
# -------------------------------------------------------------------------
class BarrierSolver:
    """Клас для розрахунку хвильової функції для сходинки та прямокутного бар'єра."""
    def __init__(self, m):
        self.m = float(m)
    def solve_step(self, E, U0, x):
        x = np.array(x, dtype=float)
        k1 = get_k(E, self.m, 0.0)
        k1 = complex(k1)
        if E > U0:
            k2 = get_k(E, self.m, U0)
            k2 = complex(k2)
            R_amp = (k1 - k2) / (k1 + k2)
            T_amp = 2.0 * k1 / (k1 + k2)
            psi = np.zeros_like(x, dtype=complex)
            left_mask = x < 0
            right_mask = x >= 0
            psi[left_mask] = np.exp(1j * k1 * x[left_mask]) + R_amp * np.exp(-1j * k1 * x[left_mask])
            psi[right_mask] = T_amp * np.exp(1j * k2 * x[right_mask])
            k1_r = k1.real if abs(k1.real) > 1e-18 else 1e-18
            k2_r = k2.real if abs(k2.real) > 1e-18 else 1e-18
            T = (k2_r / k1_r) * (abs(T_amp)**2)
            R = abs(R_amp)**2
            return np.real(psi), np.abs(psi)**2, T, R
        else:
            k2 = get_k(E, self.m, U0)
            kappa = abs(complex(k2).imag)
            psi = np.zeros_like(x, dtype=complex)
            left_mask = x < 0
            right_mask = x >= 0
            psi[left_mask] = np.exp(1j * k1 * x[left_mask]) + np.exp(-1j * k1 * x[left_mask])
            psi[right_mask] = np.exp(-kappa * x[right_mask])
            return np.real(psi), np.abs(psi)**2, 0.0, 1.0

    def solve_rectangular(self, E, U0, L, x):
        x = np.array(x, dtype=float)
        k1 = complex(get_k(E, self.m, 0.0))
        k2_complex = complex(get_k(E, self.m, U0))
        T = 0.0
        R = 1.0
        try:
            if E > U0:
                k2r = k2_complex.real
                denom = 1.0 + (U0**2 * (np.sin(k2r * L)**2)) / (4.0 * E * (E - U0))
                if denom == 0:
                    T = 0.0
                else:
                    T = 1.0 / denom
            else:
                kappa = abs(k2_complex.imag)
                if kappa * L > 100.0:
                    T = 0.0
                else:
                    denom = 1.0 + (U0**2 * (np.sinh(kappa * L)**2)) / (4.0 * E * (U0 - E))
                    T = 1.0 / denom
            R = max(0.0, 1.0 - T)
        except Exception:
            T = 0.0
            R = 1.0
        psi = np.zeros_like(x, dtype=complex)
        left_mask = x < 0
        mid_mask = (x >= 0) & (x <= L)
        right_mask = x > L
        try:
            k1c = k1
            k2c = k2_complex
            denom_t = 2.0 * k1c * k2c * np.cos(k2c * L) - 1j * (k1c**2 + k2c**2) * np.sin(k2c * L)
            if np.abs(denom_t) < 1e-16:
                t_amp = 0.0
                r_amp = 1.0
            else:
                t_amp = (2.0 * k1c * k2c * np.exp(-1j * k1c * L)) / denom_t
                r_amp = (1j * (k2c**2 - k1c**2) * np.sin(k2c * L)) / denom_t
        except Exception:
            t_amp = 0.0
            r_amp = 1.0
        if np.any(left_mask):
            psi[left_mask] = np.exp(1j * k1 * x[left_mask]) + r_amp * np.exp(-1j * k1 * x[left_mask])
        if np.any(mid_mask):
            try:
                x0 = 0.0
                xL = L
                M = np.array([
                    [np.exp(1j * k2c * x0), np.exp(-1j * k2c * x0)],
                    [1j * k2c * np.exp(1j * k2c * x0), -1j * k2c * np.exp(-1j * k2c * x0)]
                ], dtype=complex)
                psi_left_0 = 1.0 + r_amp
                psi_left_der_0 = 1j * k1 * (1.0 - r_amp)
                b = np.array([psi_left_0, psi_left_der_0], dtype=complex)
                sol = linalg.solve(M, b)
                Acoef, Bcoef = sol[0], sol[1]
            except Exception:
                Acoef, Bcoef = 0.0, 0.0
            psi[mid_mask] = Acoef * np.exp(1j * k2c * x[mid_mask]) + Bcoef * np.exp(-1j * k2c * x[mid_mask])
        if np.any(right_mask):
            psi[right_mask] = t_amp * np.exp(1j * k1 * x[right_mask])
        psi_real = np.real(psi)
        prob = np.abs(psi)**2
        return psi_real, prob, T, R

# -------------------------------------------------------------------------
# ДОПОМІЖНІ ФУНКЦІЇ (залишено тільки те, що потрібно для бар’єру)
# -------------------------------------------------------------------------
def safe_sqrt_complex(x):
    return np.sqrt(x + 0j)

def get_k(E, m, U=0.0):
    val = 2.0 * m * (E - U)
    return safe_sqrt_complex(val) / HBAR

# -------------------------------------------------------------------------
# ОСНОВНА ФУНКЦІЯ — ТІЛЬКИ БАР'ЄР
# -------------------------------------------------------------------------
def run_barrier_sim():
    st.title("Потенціальний Бар’єр та Сходинка")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        barrier_type = st.radio("Тип", ["Прямокутний бар’єр", "Сходинка"])
        E_ev = st.slider("Енергія E (еВ)", 1.0, 200.0, 40.0)
        U0_ev = st.slider("Висота U₀ (еВ)", 10.0, 300.0, 100.0)
        L_nm = st.slider("Ширина L (нм)", 0.5, 10.0, 2.0) if barrier_type == "Прямокутний бар’єр" else None

    with col2:
        E = E_ev * EV
        U0 = U0_ev * EV
        L = L_nm * 1e-9 if L_nm else 1e-9
        solver = BarrierSolver(M_E)

        if barrier_type == "Прямокутний бар’єр":
            x = np.linspace(-2*L, 6*L, 1500)
            psi_r, prob, T, R = solver.solve_rectangular(E, U0, L, x)
        else:
            x = np.linspace(-5e-9, 5e-9, 1500)
            psi_r, prob, T, R = solver.solve_step(E, U0, x)

        st.metric("T (проходження)", f"{T:.6e}")
        st.metric("R (відбиття)", f"{R:.6f}")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x*1e9, np.zeros_like(x) + E/EV, 'r--', lw=2, label=f"E = {E_ev:.1f} еВ")
        ax.plot(x*1e9, psi_r*50 + E/EV, 'cyan', lw=2, label="Re ψ(x)")
        ax.plot(x*1e9, prob*100 + E/EV, 'lime', lw=2, ls=':', alpha=0.8, label="|ψ|²")
        ax.set_xlabel("x (нм)")
        ax.set_title(barrier_type)
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    run_barrier_sim()