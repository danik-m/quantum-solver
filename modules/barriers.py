# modules/barriers.py
# Модуль для симуляції квантового бар'єра
# Експортує функцію run_barrier_sim()
# ------------------------------------------------------------------------------

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from scipy import constants
from scipy import linalg

# -------------------------------------------------------------------------
# 1. КОНСТАНТИ
# -------------------------------------------------------------------------
HBAR = constants.hbar
M_E = constants.m_e
M_P = constants.m_p
EV = constants.electron_volt

# Примітка: st.set_page_config видалено, оскільки цей файл імпортується як модуль.
# Конфігурація сторінки має бути у вашому головному файлі.

plt.style.use('default')

# -------------------------------------------------------------------------
# 2. МАТЕМАТИЧНЕ ЯДРО
# -------------------------------------------------------------------------

def safe_sqrt_complex(x):
    """Безпечний корінь для скаляра чи масиву (повертає комплексні значення при потребі)."""
    return np.sqrt(x + 0j)

def get_k(E, m, U=0.0):
    """
    Розраховує хвильовий вектор k (може бути комплексним).
    Повертає комплексне число.
    Формула: k = sqrt(2 m (E - U)) / hbar
    """
    val = 2.0 * m * (E - U)
    return safe_sqrt_complex(val) / HBAR

class BarrierSolver:
    """Клас для розрахунку хвильової функції для сходинки та прямокутного бар'єра."""
    def __init__(self, m):
        self.m = float(m)

    def solve_step(self, E, U0, x):
        """
        Розв'язок для потенціальної сходинки в x=0: U(x<0)=0, U(x>=0)=U0
        Повертає: psi_real (масив), prob_density (масив), T (float), R (float)
        """
        x = np.array(x, dtype=float)
        k1 = get_k(E, self.m, 0.0)
        k1 = complex(k1)

        if E > U0:
            k2 = get_k(E, self.m, U0)
            k2 = complex(k2)
            # Амплітуди відбиття і пропускання
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
            # E < U0
            k2 = get_k(E, self.m, U0)
            kappa = abs(complex(k2).imag)
            psi = np.zeros_like(x, dtype=complex)
            left_mask = x < 0
            right_mask = x >= 0
            
            psi[left_mask] = np.exp(1j * k1 * x[left_mask]) + np.exp(-1j * k1 * x[left_mask])
            psi[right_mask] = np.exp(-kappa * x[right_mask])
            
            # T = 0, R = 1
            return np.real(psi), np.abs(psi)**2, 0.0, 1.0

    def solve_rectangular(self, E, U0, L, x):
        """
        Розв'язок для прямокутного бар'єра шириною L.
        Повертає psi_real, prob_density, T, R
        """
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
# 3. ФУНКЦІЇ ВІЗУАЛІЗАЦІЇ
# -------------------------------------------------------------------------
def draw_arrow(ax, x1, x2, y, text, color='white'):
    ax.annotate('', xy=(x1, y), xytext=(x2, y), arrowprops=dict(arrowstyle='<->', color=color))
    ax.text((x1 + x2) / 2.0, y, text, ha='center', va='bottom', color=color,
            bbox=dict(facecolor='#0e1117', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.1'))

def plot_setup(ax, title, U_max):
    ax.set_title(title, color='white')
    ax.set_xlabel("x (м)", color='white')
    ax.set_ylabel("Енергія / Ψ", color='white')
    ymin = -abs(U_max) * 0.12
    ymax = abs(U_max) * 1.6 + 1e-30
    ax.set_ylim(ymin, ymax)
    ax.tick_params(colors='white')
    for spine in ['left', 'bottom', 'right', 'top']:
        ax.spines[spine].set_color('white')
    ax.set_facecolor('#0e1117')
    fig = ax.figure
    fig.patch.set_facecolor('#0e1117')

# -------------------------------------------------------------------------
# 4. ГОЛОВНА ЛОГІКА МОДУЛЯ (ВИПРАВЛЕНА СИГНАТУРА)
# -------------------------------------------------------------------------
def run_barrier_sim(input_params=None, input_sub_type=None):
    """
    Ця функція викликається з головного файлу.
    Параметри input_params і input_sub_type є необов'язковими.
    Це виправляє помилку 'takes 0 positional arguments but 2 were given'.
    """
    show_results = False
    
    # Якщо параметри не передані (автономний запуск або перший вхід)
    if input_params is None:
        if 'run_calc_barrier' not in st.session_state:
            st.session_state['run_calc_barrier'] = False

        st.sidebar.title("🎛 Панель Керування (Бар'єр)")

        sys_type = "Потенціальний Бар'єр"
        sub_type = st.sidebar.radio("Тип бар'єру:", ["Сходинка", "Прямокутний бар'єр"])

        st.sidebar.markdown("---")
        st.sidebar.header("Параметри")

        params = {}

        # Частинка
        particle_name = st.sidebar.selectbox("Частинка:", ["Електрон", "Протон", "Мюон"])
        mass_map = {"Електрон": M_E, "Протон": M_P, "Мюон": M_E * 206.768}
        params['m'] = float(mass_map[particle_name])

        # Параметри L
        if sub_type == "Прямокутний бар'єр":
            params['L'] = st.sidebar.number_input("Ширина L (м)", value=1e-20, step=1e-10, format="%.2e")
        else:
            params['L'] = 0.0 

        # Потенціал U0, енергія E
        params['U0'] = st.sidebar.number_input("Потенціал U₀ (Дж)", value=50.0 * EV, step=1.6e-20, format="%.2e")
        params['E'] = st.sidebar.number_input("Енергія E (Дж)", value=5.0 * EV, step=1.6e-20, format="%.2e")

        st.sidebar.markdown("---")
        if st.sidebar.button("🚀 РОЗРАХУВАТИ"):
            st.session_state['run_calc_barrier'] = True
            
        show_results = st.session_state.get('run_calc_barrier', False)
        
    else:
        # Режим виклику з main.py з переданими параметрами
        params = input_params
        sub_type = input_sub_type
        show_results = True

    # Відображення результатів
    if show_results:
        st.title(f"Результати: {sub_type}")
        m = params.get('m', M_E)
        L = params.get('L', 1e-9)
        U0 = params.get('U0', 0.0)
        E = params.get('E', 0.0)

        # ------------------------------------------------------------------
        # 1. СХОДИНКА
        # ------------------------------------------------------------------
        if sub_type == "Сходинка":
            m_val = m
            E_val = E
            U0_val = U0
            x_viz = np.linspace(-2e-9, 2e-9, 1000)

            solver = BarrierSolver(m_val)
            psi_real, psi_prob, T, R = solver.solve_step(E_val, U0_val, x_viz)

            # Виводимо метрики
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("T (Пропускання)", f"{T:.6f}")
                st.metric("R (Відбиття)", f"{R:.6f}")

            with col2:
                fig, ax = plt.subplots(figsize=(10, 5))
                plot_setup(ax, "Потенціальна Сходинка", max(E_val, U0_val, 1e-20))
                U_viz = np.where(x_viz > 0, U0_val, 0.0)
                ax.plot(x_viz, U_viz, 'w-', lw=2, label='U(x)')
                ax.axhline(E_val, color='r', ls='--', label='E')

                # Нормалізація для малювання
                if np.max(np.abs(psi_real)) > 0:
                    psi_plot = E_val + psi_real / np.max(np.abs(psi_real)) * (abs(E_val) + 0.5 * abs(U0_val) + 1e-20)
                else:
                    psi_plot = E_val + psi_real

                ax.plot(x_viz, psi_plot, color='cyan', label=r'Re($\Psi$)')
                ax.plot(x_viz, E_val + psi_prob / (np.max(psi_prob) + 1e-30) * (abs(E_val) + 0.5 * abs(U0_val) + 1e-20),
                        color='green', ls=':', label=r'$|\Psi|^2$')

                ax.legend(loc='upper right')
                st.pyplot(fig)

        # ------------------------------------------------------------------
        # 2. ПРЯМОКУТНИЙ БАР'ЄР
        # ------------------------------------------------------------------
        elif sub_type == "Прямокутний бар'єр":
            m_val = m
            E_val = E
            U0_val = U0
            L_val = L

            solver = BarrierSolver(m_val)
            x = np.linspace(-2.0 * L_val, 3.0 * L_val, 1200)
            psi_real, psi_prob, T, R = solver.solve_rectangular(E_val, U0_val, L_val, x)

            # Метрики
            st.metric("T (Пропускання)", f"{T:.6e}")
            st.metric("R (Відбиття)", f"{R:.6f}")

            # Візуалізація
            fig, ax = plt.subplots(figsize=(11, 6))
            plot_setup(ax, "Прямокутний Бар'єр", max(E_val, U0_val, 1e-20))
            U_viz = np.zeros_like(x)
            U_viz[(x >= 0) & (x <= L_val)] = U0_val
            ax.plot(x, U_viz, 'w-', lw=2, label='U(x)')
            ax.axhline(E_val, color='r', ls='--', label='E')

            # Нормалізація хвилі
            if np.max(np.abs(psi_real)) > 0:
                psi_plot = E_val + psi_real / np.max(np.abs(psi_real)) * (max(U0_val, E_val) * 0.4 + 1e-20)
            else:
                psi_plot = E_val + psi_real

            if np.max(psi_prob) > 0:
                prob_plot = E_val + psi_prob / np.max(psi_prob) * (max(U0_val, E_val) * 0.4 + 1e-20)
            else:
                prob_plot = E_val + psi_prob

            ax.plot(x, psi_plot, color='cyan', alpha=0.85, label=r'Re($\Psi$)')
            ax.plot(x, prob_plot, color='lime', ls=':', label=r'$|\Psi|^2$')
            draw_arrow(ax, 0.0, L_val, U0_val * 1.05, "L")
            ax.legend(loc='upper right')
            st.pyplot(fig)

    elif input_params is None:
        st.title("Квантовий Бар'єр")
        st.markdown("Налаштуйте параметри зліва та натисніть **🚀 РОЗРАХУВАТИ**.")

if __name__ == "__main__":
    # Для тестування можна запустити як скрипт
    st.set_page_config(layout="wide", page_title="Тест Модуля Бар'єрів")
    run_barrier_sim()