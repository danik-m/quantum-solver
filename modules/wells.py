import streamlit as st
import numpy as np
import matplotlib
# Встановлюємо бекенд Agg для стабільності у Streamlit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import constants
from scipy import optimize

# -------------------------------------------------------------------------
# 1. КОНСТАНТИ
# -------------------------------------------------------------------------
HBAR = constants.hbar
M_E = constants.m_e
M_P = constants.m_p
EV = constants.electron_volt

# Конфігурація сторінки тільки при прямому запуску
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Квантова Яма: Симуляція")

# Темна тема для графіків
plt.style.use('dark_background')

# -------------------------------------------------------------------------
# 2. МАТЕМАТИЧНЕ ЯДРО
# -------------------------------------------------------------------------

# ЗАМЕНИТЕ ЭТУ ФУНКЦИЮ НА СУЩЕСТВУЮЩУЮ В ВАШЕМ ФАЙЛЕ
def solve_finite_well_energies(m, L, U0):
    """
    Надійний та швидкий пошук рівнів енергії кінцевої ями.
    Працює навіть при L ≈ 0.03 м або U₀ ≈ 10⁴ еВ.
    """
    if U0 <= 0 or L <= 0:
        return []

    # Безрозмірний параметр η = L/2 * √(2mU₀)/ℏ
    # (в літературі часто позначають z₀)
    try:
        eta = L/2.0 * np.sqrt(2.0 * m * U0) / HBAR
    except (OverflowError, ValueError):
        return []                                 # надто великі числа → рівнів немає

    # Якщо η дуже маленьке — точно немає зв’язаних станів
    if eta < 1e-6:
        return []

    # Обмежуємо кількість рівнів розумним числом (максимум ~200)
    # Це найважливіше виправлення — без нього цикл стає нескінченним
    max_n = min(200, int(np.ceil(eta / (np.pi/2.0))) + 5)

    roots_z = []
    eps = 1e-8

    # Рівняння для парних станів
    def even_eq(z):  
        inside = np.clip((eta/z)**2 - 1.0, 0.0, None)
        return np.tan(z) - np.sqrt(inside)

    # Рівняння для непарних станів
    def odd_eq(z):
        inside = np.clip((eta/z)**2 - 1.0, 0.0, None)
        return -1.0/np.tan(z) - np.sqrt(inside)

    for n in range(max_n):
        a = n * np.pi/2.0 + eps
        b = (n + 1) * np.pi/2.0 - eps

        if a >= eta:
            break
        if b > eta:
            b = eta - eps

        if a >= b:
            continue

        # парні стани
        try:
            fa, fb = even_eq(a), even_eq(b)
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0:
                root = optimize.brentq(even_eq, a, b, xtol=1e-12, maxiter=100)
                if 0 < root < eta:
                    roots_z.append(('even', root))
        except Exception:
            pass

        # непарні стани
        try:
            fa, fb = odd_eq(a), odd_eq(b)
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0:
                root = optimize.brentq(odd_eq, a, b, xtol=1e-12, maxiter=100)
                if 0 < root < eta:
                    roots_z.append(('odd', root))
        except Exception:
            pass

    # Перетворюємо знайдені z у енергію
    energies = []
    for parity, z in roots_z:
        E = (HBAR ** 2 * (2.0 * z / L) ** 2) / (2.0 * m)     # E = ℏ²k²/2m, k = 2z/L
        if E < U0:
            kappa = np.sqrt(2.0 * m * (U0 - E)) / HBAR
            energies.append({
                'E': float(E),
                'k': 2.0 * z / L,
                'kappa': float(kappa),
                'parity': parity
            })

    return sorted(energies, key=lambda x: x['E'])

def solve_inf_well_energies(m, L, n_max=10):
    """Аналітичний розрахунок для нескінченної ями (Дж)."""
    energies = []
    for n in range(1, n_max + 1):
        E = (n**2 * np.pi**2 * HBAR**2) / (2.0 * m * L**2)
        energies.append({'E': E, 'n': n, 'parity': 'even' if n % 2 != 0 else 'odd'})
    return energies

def get_wavefunction_finite(x_math, energy_data, L):
    """
    Хвильова функція кінцевої ями (x_math центровано в 0).
    Використовує стабільну формулу для уникнення переповнення експоненти.
    """
    psi = np.zeros_like(x_math)
    k, kappa, parity = energy_data['k'], energy_data['kappa'], energy_data['parity']
    a = L / 2.0 
    
    # Замість розрахунку величезного A_out = trig * exp(kappa*a),
    # ми обчислюємо значення одразу з компенсуючою експонентою exp(-kappa*x).
    # Формула зовні: trig(ka) * exp(kappa * (a - |x|))
    # Оскільки |x| > a, то (a - |x|) < 0, тому експонента завжди мала і безпечна.
    
    val_edge = np.cos(k*a) if parity == 'even' else np.sin(k*a)

    for i, x in enumerate(x_math):
        if abs(x) <= a:
            # Всередині ями
            psi[i] = np.cos(k*x) if parity == 'even' else np.sin(k*x)
        else:
            # Зовні ями (безпечний розрахунок)
            exponent = kappa * (a - abs(x))
            # Захист від занадто малих значень (underflow), хоча для float це не критично
            if exponent < -700: 
                val = 0.0
            else:
                val = val_edge * np.exp(exponent)
            
            psi[i] = val if (parity == 'even' or x > 0) else -val
            
    norm = np.sqrt(np.trapz(psi**2, x_math))
    return psi / norm if norm > 0 else psi

def get_wavefunction_inf(x_plot, n, L):
    """Хвильова функція нескінченної ями (x_plot від 0 до L)."""
    psi = np.zeros_like(x_plot)
    mask = (x_plot >= 0) & (x_plot <= L)
    psi[mask] = np.sqrt(2.0 / L) * np.sin(n * np.pi * x_plot[mask] / L)
    return psi

# -------------------------------------------------------------------------
# 3. ФУНКЦІЇ ВІЗУАЛІЗАЦІЇ
# -------------------------------------------------------------------------

def setup_plot_style(ax, title, xlabel="x (м)", ylabel="Енергія (еВ)"):
    """Базове налаштування стилю графіка."""
    ax.set_title(title, color='white', fontsize=16, pad=20)
    ax.set_xlabel(xlabel, color='white', fontsize=12)
    ax.set_ylabel(ylabel, color='white', fontsize=12)
    ax.tick_params(colors='white', labelsize=10)
    ax.set_facecolor('#0E1117') # Темний фон, як у Streamlit
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.grid(True, linestyle='--', alpha=0.2, color='white')

def draw_level_and_wave(ax, x, psi, prob, E_ev, L, color_psi='cyan', color_prob='lime'):
    """Малює рівень енергії та хвильову функцію поверх нього."""
    
    # Визначаємо масштаб хвилі.
    scale_factor = E_ev * 0.3 if E_ev > 1e-3 else 1.0
    
    # Зсуваємо хвилю на рівень енергії E
    psi_shifted = E_ev + psi * scale_factor
    prob_shifted = E_ev + prob * scale_factor
    
    # 1. Лінія енергії (пунктир)
    ax.hlines(E_ev, x[0], x[-1], colors='red', linestyles='--', linewidth=1, alpha=0.7)
    
    # 2. Підпис енергії справа
    ax.text(x[-1], E_ev, f"  E = {E_ev:.3e} еВ", 
            color='red', va='center', fontsize=11, fontweight='bold')
    
    # 3. Хвильова функція (Лінія)
    ax.plot(x, psi_shifted, color=color_psi, linewidth=2, label=r'$\Psi(x)$')
    
    # 4. Заливка під хвилею (напівпрозора)
    ax.fill_between(x, E_ev, psi_shifted, color=color_psi, alpha=0.15)
    
    # 5. Густина ймовірності (Пунктир)
    ax.plot(x, prob_shifted, color=color_prob, linestyle=':', linewidth=2, label=r'$|\Psi|^2$')

# -------------------------------------------------------------------------
# 4. ЕКСПОРТОВАНІ ФУНКЦІЇ СИМУЛЯЦІЇ
# -------------------------------------------------------------------------

def run_finite_well_sim(params):
    """
    Симуляція Кінцевої ями.
    params: {'m': float, 'L': float, 'U0': float (Дж)}
    """
    m = params.get('m', M_E)
    L_val = params.get('L', 1e-9)
    U0_joule = params.get('U0', 10.0 * EV)
    U0_ev = U0_joule / EV

    st.subheader("📦 Кінцева Потенціальна Яма")

    energies = solve_finite_well_energies(m, L_val, U0_joule)
    
    if not energies:
        st.warning(f"⚠️ При глибині {U0_ev:.2f} еВ та ширині {L_val:.2e} м зв'язаних станів не знайдено.")
        return

    # Слайдер вибору n
    n = st.slider("Квантове число n", 1, 10, 1, key="infinite_n_slider")
    
    state = energies[n-1]
    E_ev = state['E'] / EV
    parity_str = "Парна (Even)" if state['parity']=='even' else "Непарна (Odd)"

    # Метрики
    c1, c2, c3 = st.columns(3)
    c1.metric("Рівень", f"n = {n}")
    c2.metric("Енергія", f"{E_ev:.4e} еВ")
    c3.metric("Симетрія", parity_str)

    # Графік
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0E1117')
    setup_plot_style(ax, f"Стаціонарний стан n={n}")

    # Діапазон X: від -L до 2L (щоб показати хвости)
    x = np.linspace(-L_val*0.8, L_val*1.8, 1000)
    
    # 1. Малюємо Яму (Стінки)
    ax.fill_between(x, 0, U0_ev, where=(x <= 0), color='#4A90E2', alpha=0.2, label='Стінки бар\'єру')
    ax.fill_between(x, 0, U0_ev, where=(x >= L_val), color='#4A90E2', alpha=0.2)
    
    # Лінія потенціалу
    U_pot = np.where((x >= 0) & (x <= L_val), 0, U0_ev)
    ax.plot(x, U_pot, color='white', linewidth=2)
    ax.text(0, U0_ev * 1.02, f" U₀ = {U0_ev:.1f} еВ", color='white', fontsize=10)

    # 2. Розрахунок хвилі
    x_math = x - L_val/2.0
    psi = get_wavefunction_finite(x_math, state, L_val)
    prob = psi**2
    if np.max(np.abs(psi)) > 0:
        psi /= np.max(np.abs(psi))
        prob /= np.max(prob)

    # 3. Малюємо хвилю і рівень
    draw_level_and_wave(ax, x, psi, prob, E_ev, L_val)

    # Налаштування меж осей
    y_max_plot = max(U0_ev * 1.3, E_ev * 1.5)
    ax.set_ylim(-y_max_plot * 0.1, y_max_plot)
    ax.set_xlim(x[0], x[-1])

    ax.legend(loc='upper right', facecolor='#262730', labelcolor='white')
    st.pyplot(fig)
    
    # ТЕОРІЯ
    st.markdown("---")
    st.markdown("""
    ### 📝 Пояснення
    У кінцевій ямі стінки мають висоту $U_0$. Частинка не заперта ідеально:
    * **Всередині ($0 < x < L$):** Хвильова функція осцилює (sin/cos).
    * **В стінках ($x < 0, x > L$):** Енергія частинки $E < U_0$, тому кінетична енергія формально від'ємна. Хвильова функція експоненційно затухає (**Туннельний ефект**).
    """)

def run_infinite_well_sim(params):
    """
    Симуляція Нескінченної ями.
    params: {'m': float, 'L': float}
    """
    m = params.get('m', M_E)
    L_val = params.get('L', 1e-9)

    st.subheader("📦 Нескінченна Потенціальна Яма")

    energies = solve_inf_well_energies(m, L_val, n_max=10)
    E_max_limit = energies[-1]['E'] / EV

    # Слайдер вибору n
    n = st.slider("Квантове число n", 1, 10, 1, key="infinite_n_slider")
    
    state = energies[n-1]
    E_ev = state['E'] / EV
    parity_str = "Парна (Even)" if n % 2 != 0 else "Непарна (Odd)"

    c1, c2, c3 = st.columns(3)
    c1.metric("Рівень", f"n = {n}")
    c2.metric("Енергія", f"{E_ev:.4e} еВ")
    c3.metric("Симетрія", parity_str)

    # Графік
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0E1117')
    setup_plot_style(ax, f"Стаціонарний стан n={n}")

    # Діапазон X
    x = np.linspace(-L_val*0.2, L_val*1.2, 1000)
    
    # 1. Малюємо Стінки
    ax.axvline(0, color='white', linewidth=3)
    ax.axvline(L_val, color='white', linewidth=3)
    
    ymax_fill = E_max_limit * 1.5
    ax.fill_between(x, -ymax_fill, ymax_fill, where=(x<0), color='gray', alpha=0.3, hatch='//')
    ax.fill_between(x, -ymax_fill, ymax_fill, where=(x>L_val), color='gray', alpha=0.3, hatch='//')
    ax.hlines(0, 0, L_val, color='white', linewidth=1)

    # 2. Розрахунок хвилі
    psi = get_wavefunction_inf(x, n, L_val)
    prob = psi**2
    if np.max(np.abs(psi)) > 0:
        psi /= np.max(np.abs(psi))
        prob /= np.max(prob)

    # 3. Малюємо хвилю
    draw_level_and_wave(ax, x, psi, prob, E_ev, L_val)

    # Налаштування меж
    ax.set_ylim(-E_max_limit * 0.1, E_max_limit * 1.2)
    ax.set_xlim(x[0], x[-1])

    ax.legend(loc='upper right', facecolor='#262730', labelcolor='white')
    st.pyplot(fig)
    
    # ТЕОРІЯ
    st.markdown("---")
    st.markdown("""
    ### 📝 Пояснення
    У нескінченній ямі стінки абсолютно непроникні.
    * **Граничні умови:** $\Psi(0) = 0$ та $\Psi(L) = 0$.
    * **Енергія:** $E_n \\sim n^2$. Відстань між рівнями швидко зростає.
    * Хвильова функція строго локалізована в межах $0..L$.
    """)

# -------------------------------------------------------------------------
# 5. MAIN — ГОТОВИЙ ДЛЯ ВСТАВКИ
# -------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Квантова Яма", layout="wide")
    st.title("Квантова потенціальна яма")

    st.sidebar.header("Налаштування симуляції")

    # 1. Частинка
    st.sidebar.subheader("Частинка")
    particle = st.sidebar.radio(
        "Оберіть частинку",
        options=["Електрон", "Мюон", "Протон"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    mass_map = {"Електрон": M_E, "Мюон": 206.768 * M_E, "Протон": M_P}
    m = mass_map[particle]

    # 2. Довжина ями
    st.sidebar.subheader("Довжина ями")
    L_nm = st.sidebar.slider(
        "L (нм)",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.1,
        format="%.3f"
    )
    L_meters = L_nm * 1e-9
    st.sidebar.markdown(f"**L = {L_nm:.3f} нм**")

    # 3. Глибина потенціалу (тільки для кінцевої ями)
    st.sidebar.subheader("Потенціал бар’єру")
    U0_ev = st.sidebar.number_input(
        "U₀ (еВ)",
        min_value=0.1,
        value=50.0,
        step=1.0,
        help="Глибина ями для кінцевої моделі"
    )

    # Вибір типу ями
    well_type = st.sidebar.radio("Тип ями", ["Нескінченна яма", "Кінцева яма"])

    # ЗАПУСК СИМУЛЯЦІЇ — ОЦЕ САМЕ ГОЛОВНЕ!
    if well_type == "Кінцева яма":
        run_finite_well_sim(m, L_meters, U0_ev * EV)
    else:
        run_infinite_well_sim(m, L_meters)


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()