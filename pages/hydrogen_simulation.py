import streamlit as st
import numpy as np
import matplotlib
# Встановлюємо бекенд Agg перед імпортом pyplot, щоб уникнути помилок потоків GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import constants
from scipy import optimize

# -------------------------------------------------------------------------
# 1. КОНФИГУРАЦИЯ И КОНСТАНТЫ
# -------------------------------------------------------------------------
HBAR = constants.hbar
M_E = constants.m_e
M_P = constants.m_p
EV = constants.electron_volt

st.set_page_config(layout="wide", page_title="Потенциальная Яма: Конечная и Бесконечная")

# -------------------------------------------------------------------------
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ВИЗУАЛИЗАЦИИ
# -------------------------------------------------------------------------
def draw_arrow(ax, x1, x2, y, text, color='white'):
    """Рисует стрелку размера."""
    ax.annotate('', xy=(x1, y), xytext=(x2, y), arrowprops=dict(arrowstyle='<->', color=color))
    ax.text((x1 + x2) / 2.0, y, text, ha='center', va='bottom', color=color,
            bbox=dict(facecolor='#0e1117', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.1'))

def plot_setup(ax, title, y_limit_ref, L):
    """Настройка осей и цветов графика."""
    ax.set_title(title, color='white', fontsize=16)
    ax.set_xlabel("x (м)", color='white', fontsize=12)
    ax.set_ylabel("Энергия (эВ)", color='white', fontsize=12)
    
    # Жесткие границы по Y для визуализации
    ymin = -abs(y_limit_ref) * 0.1
    ymax = abs(y_limit_ref) * 1.5
    ax.set_ylim(ymin, ymax)
    
    # Расширяем границы по X
    ax.set_xlim(-L * 0.5, L * 1.8)
    
    ax.tick_params(colors='white', which='both')
    for spine in ['left', 'bottom', 'right', 'top']:
        ax.spines[spine].set_color('white')
    
    ax.set_facecolor('#0e1117')
    fig = ax.figure
    fig.patch.set_facecolor('#0e1117')

# -------------------------------------------------------------------------
# 3. МАТЕМАТИЧЕСКОЕ ЯДРО (КОНЕЧНАЯ ЯМА)
# -------------------------------------------------------------------------
def solve_finite_well_energies(m, L, U0):
    """Находит уровни энергии для КОНЕЧНОЙ ямы методом Брента."""
    if U0 <= 0 or L <= 0:
        return []

    # P - параметр мощности ямы
    P = (np.sqrt(2 * m * U0) * L) / (2 * HBAR)
    max_z = P 
    
    roots_z = []
    
    def eq_even(z):
        term = np.maximum((P/z)**2 - 1, 0)
        return np.tan(z) - np.sqrt(term)
    
    def eq_odd(z):
        term = np.maximum((P/z)**2 - 1, 0)
        return -1.0/np.tan(z) - np.sqrt(term)

    n_levels = int(1 + np.floor(2 * P / np.pi))
    
    for n in range(n_levels):
        lower = n * (np.pi / 2.0) + 1e-4
        upper = min((n + 1) * (np.pi / 2.0) - 1e-4, max_z - 1e-6)
        
        if lower >= upper: continue
            
        try:
            if n % 2 == 0: # Четные (Even)
                res = optimize.brentq(eq_even, lower, upper)
                roots_z.append((res, 'even'))
            else: # Нечетные (Odd)
                res = optimize.brentq(eq_odd, lower, upper)
                roots_z.append((res, 'odd'))
        except ValueError:
            pass 
            
    energies = []
    for z, parity in roots_z:
        E = (2.0 * HBAR * z / L)**2 / (2.0 * m)
        if E < U0: 
            energies.append({
                'E': E,
                'z': z,
                'parity': parity,
                'k': 2.0 * z / L,
                'kappa': np.sqrt(2 * m * (U0 - E)) / HBAR
            })
    
    energies.sort(key=lambda x: x['E'])
    return energies

def get_wavefunction_finite(x_plot, energy_data, L):
    """Строит волновую функцию для КОНЕЧНОЙ ямы."""
    x_math = x_plot - L/2.0 # Центрируем для математики
    
    psi = np.zeros_like(x_math)
    k = energy_data['k']
    kappa = energy_data['kappa']
    parity = energy_data['parity']
    a = L / 2.0 
    
    A_in = 1.0
    
    if parity == 'even':
        val_edge = np.cos(k * a)
        A_out = val_edge * np.exp(kappa * a)
        for i, x in enumerate(x_math):
            if abs(x) <= a:
                psi[i] = A_in * np.cos(k * x)
            else:
                psi[i] = A_out * np.exp(-kappa * abs(x))
    else:
        val_edge = np.sin(k * a)
        A_out = val_edge * np.exp(kappa * a)
        for i, x in enumerate(x_math):
            if abs(x) <= a:
                psi[i] = A_in * np.sin(k * x)
            else:
                psi[i] = A_out * np.exp(-kappa * abs(x))
                if x < 0: psi[i] *= -1
                
    norm = np.sqrt(np.trapz(psi**2, x_math))
    if norm > 0:
        psi /= norm
        
    return psi

# -------------------------------------------------------------------------
# 4. МАТЕМАТИЧЕСКОЕ ЯДРО (БЕСКОНЕЧНАЯ ЯМА)
# -------------------------------------------------------------------------
def solve_inf_well_energies(m, L, n_max=10):
    """Находит уровни энергии для БЕСКОНЕЧНОЙ ямы (аналитически)."""
    energies = []
    for n in range(1, n_max + 1):
        E = (n**2 * np.pi**2 * HBAR**2) / (2.0 * m * L**2)
        # В симметричной яме n=1 (четная), n=2 (нечетная)
        parity = 'even' if n % 2 != 0 else 'odd' 
        energies.append({
            'E': E,
            'n': n,
            'parity': parity
        })
    return energies

def get_wavefunction_inf(x_plot, n, L):
    """Строит волновую функцию для БЕСКОНЕЧНОЙ ямы."""
    psi = np.zeros_like(x_plot)
    mask = (x_plot >= 0) & (x_plot <= L)
    psi[mask] = np.sqrt(2.0 / L) * np.sin(n * np.pi * x_plot[mask] / L)
    return psi

# -------------------------------------------------------------------------
# 5. ИНТЕРФЕЙС И ЛОГИКА
# -------------------------------------------------------------------------
def main():
    st.sidebar.title("🎛 Панель Управления")
    st.sidebar.header("1. Частица")

    particle_name = st.sidebar.selectbox("Выберите частицу:", ["Электрон", "Протон", "Мюон"])
    mass_map = {"Электрон": M_E, "Протон": M_P, "Мюон": M_E * 206.768}
    m = mass_map[particle_name]

    st.sidebar.header("2. Параметры Ямы")
    
    # Выбор типа ямы
    well_type = st.sidebar.radio("Тип стенок:", ["Конечные стенки", "Бесконечные стенки"])
    
    L_val = st.sidebar.number_input("Ширина L (м)", value=1e-9, step=1e-10, format="%.2e")
    
    # Потенциал только для конечной ямы
    if well_type == "Конечные стенки":
        U0_val_ev = st.sidebar.number_input("Потенциал U₀ (эВ)", value=10.0, step=0.1, format="%.2f")
        U0_val = U0_val_ev * EV
    else:
        U0_val_ev = None
        U0_val = None

    st.sidebar.markdown("---")
    
    # --- БЛОК 1: ЗАГОЛОВОК И ПАРНОСТЬ ---
    title_text = "Конечная Потенциальная Яма" if well_type == "Конечные стенки" else "Бесконечная Потенциальная Яма"
    st.title(f"📦 {title_text}")

    with st.container():
        st.markdown("""
        ### 🌗 Что такое Парность (Parity)?
        В квантовой механике, если потенциал симметричен ($U(x) = U(-x)$), волновые функции имеют определенную **парность**:
        
        * **Парная (Четная / Even) (+):** Функция симметрична, $\Psi(-x) = \Psi(x)$.
        * **Непарная (Нечетная / Odd) (-):** Функция антисимметрична, $\Psi(-x) = -\Psi(x)$.
        """)
        st.info("Уровни энергии всегда чередуются: четный, нечетный, четный...")

    # --- ЛОГИКА РАСЧЕТА И ВИЗУАЛИЗАЦИИ ---
    
    if well_type == "Конечные стенки":
        # === ЛОГИКА КОНЕЧНОЙ ЯМЫ ===
        energies_data = solve_finite_well_energies(m, L_val, U0_val)
        
        if not energies_data:
            st.error("Связанных уровней не найдено. Попробуйте увеличить ширину ямы или потенциал.")
            return

        num_levels = len(energies_data)
        n_viz = st.slider("Квантовое число n", 1, num_levels, 1)
        
        state = energies_data[n_viz - 1]
        E_n = state['E']
        parity_str = "Четная (Even)" if state['parity'] == 'even' else "Нечетная (Odd)"
        
        # Метрики
        c1, c2, c3 = st.columns(3)
        c1.metric("Квантовое число", f"n = {n_viz}")
        c2.metric("Энергия E", f"{E_n/EV:.4f} эВ")
        c3.metric("Парность", parity_str)

        # График
        fig, ax = plt.subplots(figsize=(12, 7))
        plot_setup(ax, f"Конечная яма: n={n_viz} ({parity_str})", U0_val_ev, L_val)
        
        x_viz = np.linspace(-L_val * 0.5, L_val * 1.5, 1200)
        
        # Потенциал
        U_pot = np.where((x_viz >= 0) & (x_viz <= L_val), 0.0, U0_val_ev)
        ax.plot(x_viz, U_pot, 'w-', lw=2, alpha=0.6, label='Потенциал U(x)')
        
        # Энергия
        E_ev = E_n / EV
        ax.hlines(E_ev, x_viz[0], L_val * 1.4, colors='red', linestyles='--', linewidth=1.5)
        ax.text(L_val * 1.42, E_ev, f" $E_{n_viz} = {E_ev:.3f}$ эВ", color='red', fontsize=12, va='center', fontweight='bold')
        
        # Волна
        psi = get_wavefunction_finite(x_viz, state, L_val)
        scale = U0_val_ev * 0.25
        
        psi_norm = psi / np.max(np.abs(psi)) if np.max(np.abs(psi)) > 0 else psi
        psi_plot = E_ev + psi_norm * scale
        prob_plot = E_ev + (psi_norm**2) * scale
        
        ax.plot(x_viz, psi_plot, color='cyan', lw=2.5, label=r'$\Psi_n(x)$')
        ax.fill_between(x_viz, E_ev, psi_plot, color='cyan', alpha=0.2)
        ax.plot(x_viz, prob_plot, color='lime', linestyle=':', lw=1.5, alpha=0.7, label=r'$|\Psi|^2$')
        
        draw_arrow(ax, 0, L_val, -U0_val_ev * 0.05, f"L = {L_val:.1e} м")
        ax.legend(loc='upper right', facecolor='#0e1117', labelcolor='white', framealpha=0.8)
        st.pyplot(fig)
        
        # Теория для конечной
        st.markdown("---")
        st.header("📚 Теория: Конечная яма")
        col_t, col_e = st.columns(2)
        with col_t:
            st.markdown(r"""
            **Особенности:**
            1. **Туннельный эффект:** Волновая функция проникает в стенки ($e^{-\kappa x}$).
            2. **Конечное число уровней:** Частица может покинуть яму, если $E > U_0$.
            """)
        with col_e:
            st.markdown("**Примеры:** Квантовые точки, атомное ядро, гетероструктуры.")

    else:
        # === ЛОГИКА БЕСКОНЕЧНОЙ ЯМЫ ===
        energies_data = solve_inf_well_energies(m, L_val, n_max=10)
        
        n_viz = st.slider("Квантовое число n", 1, 10, 1)
        state = energies_data[n_viz - 1]
        E_n = state['E']
        parity_str = "Четная (Even)" if n_viz % 2 != 0 else "Нечетная (Odd)"

        c1, c2, c3 = st.columns(3)
        c1.metric("Квантовое число", f"n = {n_viz}")
        c2.metric("Энергия E", f"{E_n/EV:.4f} эВ")
        c3.metric("Парность", parity_str)

        fig, ax = plt.subplots(figsize=(12, 7))
        # Для масштаба берем текущую энергию как референс (немного выше нее)
        plot_setup(ax, f"Бесконечная яма: n={n_viz}", E_n/EV * 1.5, L_val)
        
        x_viz = np.linspace(-L_val * 0.2, L_val * 1.2, 1000)
        
        # Стенки (бесконечные) - рисуем вертикальные линии
        ax.vlines([0, L_val], -E_n/EV * 0.1, E_n/EV * 2, colors='white', linewidth=3, label='Стенки ($\infty$)')
        ax.hlines(0, -L_val*0.2, L_val*1.2, color='white', lw=1)
        
        # Энергия
        E_ev = E_n / EV
        ax.hlines(E_ev, x_viz[0], L_val * 1.4, colors='red', linestyles='--', linewidth=1.5)
        ax.text(L_val * 1.42, E_ev, f" $E_{n_viz} = {E_ev:.3f}$ эВ", color='red', fontsize=12, va='center', fontweight='bold')
        
        # Волна
        psi = get_wavefunction_inf(x_viz, n_viz, L_val)
        
        scale = E_ev * 0.4
        
        psi_norm = psi / np.max(np.abs(psi)) if np.max(np.abs(psi)) > 0 else psi
        psi_plot = E_ev + psi_norm * scale
        prob_plot = E_ev + (psi_norm**2) * scale
        
        ax.plot(x_viz, psi_plot, color='cyan', lw=2.5, label=r'$\Psi_n(x)$')
        ax.fill_between(x_viz, E_ev, psi_plot, color='cyan', alpha=0.2)
        ax.plot(x_viz, prob_plot, color='lime', linestyle=':', lw=1.5, alpha=0.7, label=r'$|\Psi|^2$')
        
        draw_arrow(ax, 0, L_val, -E_ev * 0.05, f"L = {L_val:.1e} м")
        ax.legend(loc='upper right', facecolor='#0e1117', labelcolor='white', framealpha=0.8)
        st.pyplot(fig)

        # Теория для бесконечной
        st.markdown("---")
        st.header("📚 Теория: Бесконечная яма")
        col_t, col_e = st.columns(2)
        with col_t:
            st.markdown(r"""
            **Особенности:**
            1. **Идеальная модель:** Стенки непроницаемы ($U = \infty$).
            2. **Волновая функция:** Строго ноль на границах.
            3. **Энергия:** Растет квадратично $E_n \propto n^2$.
            """)
        with col_e:
            st.markdown("**Формула:** $E_n = \frac{n^2 \pi^2 \hbar^2}{2mL^2}$")

if __name__ == "__main__":
    main()
    