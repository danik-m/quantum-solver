import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import hermite, factorial

# Попытка импорта утилит (для работы и как модуль, и отдельно)
try:
    from .utils import HBAR, EV, M_E, plot_setup, draw_arrow
except ImportError:
    from utils import HBAR, EV, M_E, plot_setup, draw_arrow

# --- МАТЕМАТИКА ---

def calc_harmonic_energy(omega, n):
    """E_n = hbar * omega * (n + 0.5)"""
    return HBAR * omega * (n + 0.5)

def solve_oscillator(omega, m, n_max=10):
    """Список энергий"""
    return [calc_harmonic_energy(omega, n) for n in range(n_max + 1)]

def psi_oscillator(x, m, omega, n):
    """Хвильова функція (Ерміт)"""
    alpha = np.sqrt(m * omega / HBAR)
    xi = alpha * x
    if n > 50: n = 50
    norm_coef = 1.0 / np.sqrt((2**n) * math.factorial(n)) * np.sqrt(alpha / np.sqrt(np.pi))
    Hn = hermite(n)
    psi = norm_coef * np.exp(-0.5 * xi**2) * Hn(xi)
    return np.real(psi)

# --- ВІЗУАЛІЗАЦІЯ ---

def run_oscillator_sim(params):
    st.markdown("## 〰️ Гармонічний Осцилятор")
    
    # --- ОПИС ТА ТЕОРІЯ (НОВЕ!) ---
    with st.expander("📚 Що це таке? (Теорія та приклади)", expanded=False):
        st.markdown(r"""
        **Квантовий гармонічний осцилятор** — це одна з найважливіших моделей у квантовій механіці. Вона описує частинку, що знаходиться в параболічному потенціалі $U(x) = \frac{1}{2}m\omega^2 x^2$.
        
        ### 🔹 Основні властивості:
        1.  **Квантування енергії:** Рівні енергії розташовані на рівних відстанях:
            $$ E_n = \hbar \omega \left(n + \frac{1}{2}\right) $$
        2.  **Нульова енергія:** Навіть при $n=0$ енергія не дорівнює нулю ($E_0 = \hbar\omega/2$). Це наслідок принципу невизначеності.
        3.  **Тунелювання:** Хвильова функція проникає в класично заборонену область (за межі параболи).

        ### 🔹 Приклади в природі:
        * **Коливання атомів у молекулах** (наприклад, двоатомна молекула як пружинка).
        * **Фонони** (коливання кристалічної ґратки).
        * **Електромагнітне поле** в квантовій оптиці (фотони).
        """)

    # --- ГРАФІК ---
    omega, m = params['omega'], params['m']
    energies = solve_oscillator(omega, m, 10)
    
    n_viz = st.slider("Оберіть квантовий рівень n", 0, 5, 0, key="osc_n_slider_internal")
    E_n = energies[n_viz]
    
    st.success(f"Рівень n={n_viz}: E = {E_n:.4e} Дж ({E_n/EV:.4f} еВ)")
    
    # Темний стиль для графіка (як на скріншоті)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Колір фону самого графіка і фігури
    fig.patch.set_facecolor('#0e1117') 
    ax.set_facecolor('#0e1117')
    
    # Межі
    if m > 0 and omega > 0 and E_n > 0:
        x_turn = np.sqrt(2.0 * E_n / (m * omega**2))
        x_turn_max = np.sqrt(2.0 * energies[-1] / (m * omega**2))
    else:
        x_turn, x_turn_max = 1e-9, 1e-9
        
    x_lim = max(x_turn_max * 1.3, 1e-10)
    x = np.linspace(-x_lim, x_lim, 800)
    
    # Потенціал (Біла лінія)
    U = 0.5 * m * omega**2 * x**2
    ax.plot(x, U, color='white', linewidth=2, label='U(x)')
    
    # Хвиля
    psi = psi_oscillator(x, m, omega, n_viz)
    
    # Масштабування хвилі, щоб вона гарно виглядала на фоні енергії
    scale = (energies[1] - energies[0]) * 0.8
    psi_plot = E_n + psi / np.max(np.abs(psi)) * scale
    prob_plot = E_n + (psi**2) / np.max(psi**2) * scale

    # Лінії
    ax.plot(x, psi_plot, label=r'$\Psi$', color='cyan', linewidth=2)
    ax.plot(x, prob_plot, label=r'$|\Psi|^2$', color='magenta', linestyle=':', linewidth=2)
    
    # Заливка під квадратом модуля (пурпурна, напівпрозора)
    ax.fill_between(x, E_n, prob_plot, color='magenta', alpha=0.2)
    
    # Рівень енергії (червоний пунктир)
    ax.hlines(E_n, -x_lim, x_lim, colors='red', linestyles='--', linewidth=1, label=f'E_{n_viz}')
    
    # Стрілка ширини (2A)
    draw_arrow(ax, -x_turn, x_turn, E_n * 1.05, f"2A={2.0 * x_turn:.1e} м", color='white')

    # Налаштування осей (білі підписи)
    ax.set_xlabel("x (м)", color='white', fontsize=12)
    ax.set_ylabel("Енергія / Ψ", color='white', fontsize=12)
    ax.set_title(f"Гармонічний Осцилятор (n={n_viz})", color='white', fontsize=14)
    
    # Колір поділок
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    # Рамка (spines)
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    ax.legend(loc='upper right', facecolor='#0e1117', labelcolor='white')
    st.pyplot(fig)

# --- ГОЛОВНА ФУНКЦІЯ МОДУЛЯ ---

def main():
    st.set_page_config(page_title="Гармонічний Осцилятор", layout="wide")
    st.sidebar.header("Налаштування")
    
    from scipy import constants
    
    p_name = st.sidebar.selectbox("Частинка:", ["Електрон", "Протон", "Мюон"], key="osc_p")
    mass_map = {"Електрон": constants.m_e, "Протон": constants.m_p, "Мюон": constants.m_e * 207}
    
    params = {}
    params['m'] = float(mass_map[p_name])
    params['omega'] = st.sidebar.number_input("Частота ω (рад/с)", value=5e15, format="%.2e", step=1e14, key="osc_w")
    
    if st.sidebar.button("Розрахувати", key="osc_btn"):
        run_oscillator_sim(params)

if __name__ == "__main__":
    main()