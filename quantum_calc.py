# streamlit_quantum_simulator_fixed.py
# Виправлена та розширена версія вашого симулятора
# Працює з: Python 3.10+, streamlit, numpy, scipy, matplotlib
# Автор: виправлення / рефакторинг ChatGPT
# ------------------------------------------------------------------------------

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import time
from scipy.sparse import diags

from scipy import constants
from scipy.special import hermite
from scipy import optimize, linalg

# -------------------------------------------------------------------------
# 1. КОНФІГУРАЦІЯ ТА КОНСТАНТИ
# -------------------------------------------------------------------------
HBAR = constants.hbar
M_E = constants.m_e
M_P = constants.m_p
EV = constants.electron_volt

st.set_page_config(layout="wide", page_title="Квантовий Симулятор Ultimate")
plt.style.use('default')
PLOT_COLOR_PSI = 'blue'
PLOT_COLOR_PROB = 'red'

# -------------------------------------------------------------------------
# 2. МАТЕМАТИЧНЕ ЯДРО (PHYSICS ENGINE) - РЕФАКТОРИНГ ТА ВИПРАВЛЕННЯ
# -------------------------------------------------------------------------

def safe_sqrt_complex(x):
    """Безпечний корінь для скаляра чи масиву (повертає комплексні значення при потребі)."""
    return np.sqrt(x + 0j)

def get_k(E, m, U=0.0):
    """
    Розраховує хвильовий вектор k (може бути комплексним).
    Повертає комплексне число (або масив, якщо вхідні аргументи масиви).
    Формула: k = sqrt(2 m (E - U)) / hbar
    """
    # Перетворюємо в numpy scalar для коректних операцій
    val = 2.0 * m * (E - U)
    # Якщо val від'ємне -> отримаємо уявну хвиля (експоненційне затухання)
    return safe_sqrt_complex(val) / HBAR

def calc_inf_well_energy(m, L, n):
    """Аналітичний рівень енергії для нескінченної ями."""
    return (n**2 * np.pi**2 * HBAR**2) / (2.0 * m * L**2)

def solve_inf_well(L, m, Nlevels=5):
    """Повертає список енергій перших Nlevels для нескінченної ями."""
    energies = [calc_inf_well_energy(m, L, n+1) for n in range(Nlevels)]
    return energies

def calc_harmonic_energy(omega, n):
    """Аналітична формула для осцилятора: E_n = hbar * omega * (n + 1/2)"""
    return HBAR * omega * (n + 0.5)

def psi_inf_well(x, L, n):
    """Хвильова функція нескінченної ями (реальна)."""
    psi = np.zeros_like(x, dtype=float)
    mask = (x >= 0) & (x <= L)
    psi[mask] = np.sqrt(2.0 / L) * np.sin(n * np.pi * x[mask] / L)
    return psi

def psi_oscillator(x, m, omega, n):
    """
    Хвильова функція осцилятора (реальна при стандартному виборі).
    Використовує numpy-версію Ермітових поліномів з scipy.special.hermite.
    """
    alpha = np.sqrt(m * omega / HBAR)
    xi = alpha * x
    if n > 50:
        n = 50  # безпечне обмеження
    Hn = hermite(n)
    # Нормування
    norm_coef = 1.0 / np.sqrt((2**n) * math.factorial(n)) * np.sqrt(alpha / np.sqrt(np.pi))
    psi = norm_coef * np.exp(-0.5 * xi**2) * Hn(xi)
    # Переконаємось, що тип масиву - float
    return np.real(psi)

# ---------------------------
# Функції для кінцевої ями: чисельні рівні
# ---------------------------
def _even_equation(z, z0):
    # tan(z) = sqrt((z0/z)^2 - 1)
    # Вихід: f(z) = z * tan(z) - sqrt(z0^2 - z^2)
    lhs = z * np.tan(z)
    rhs = np.sqrt(np.maximum(z0**2 - z**2, 0.0))
    return lhs - rhs

def _odd_equation(z, z0):
    # -cot(z) = sqrt((z0/z)^2 - 1)
    # f(z) = -z * cot(z) - sqrt(z0^2 - z^2)
    lhs = -z / np.tan(z)
    rhs = np.sqrt(np.maximum(z0**2 - z**2, 0.0))
    return lhs - rhs

def finite_well_solver(m, L, U0):
    """
    Оригінальна ваша функція лишилась, зберігаю - повертає попередню оцінку кількості рівнів.
    """
    if U0 <= 0:
        return 0, 0.0
    z0 = (L / 2.0) * np.sqrt(2.0 * m * U0) / HBAR
    N = 1 + int((2.0 * z0) / np.pi)
    return N, z0

def solve_finite_well(m, L, U0, tol=1e-9, maxroots=50):
    """
    Чисельне знаходження енергій зв'язаних рівнів для симетричної прямокутної ями глибини U0 (вище навколо).
    Повертає список енергій (в джоулях), впорядкованих по зростанню.
    Використовує "z" параметризацію: z = k * L/2, z0 = (L/2)*sqrt(2 m U0)/hbar
    Розв'язує рівняння для парних/непарних станів.
    """
    if U0 <= 0 or L <= 0:
        return []

    z0 = (L / 2.0) * np.sqrt(2.0 * m * U0) / HBAR
    roots = []

    # Інтервали для z: (0, z0)
    # Парні корені: z * tan(z) = sqrt(z0^2 - z^2)  (use f_even)
    # Непарні корені: -z * cot(z) = sqrt(z0^2 - z^2) (use f_odd)
    # Знаходимо корені по інтервалах між полюсами tan/cot
    nmax = int(np.ceil(z0 / np.pi)) + 5
    for n in range(0, nmax):
        # even interval approx: around n*pi
        a = n * np.pi + 1e-6
        b = (n + 0.5) * np.pi - 1e-6
        if a < b:
            try:
                fa = _even_equation(a, z0)
                fb = _even_equation(min(b, z0 - 1e-8), z0)
                if fa * fb < 0:
                    root = optimize.brentq(lambda z: _even_equation(z, z0), a, min(b, z0 - 1e-8), maxiter=200)
                    if 0 < root < z0:
                        roots.append(root)
            except Exception:
                pass

        # odd interval approx: around (n+0.5)*pi
        a2 = (n + 0.5) * np.pi + 1e-6
        b2 = (n + 1.0) * np.pi - 1e-6
        if a2 < b2:
            try:
                fa = _odd_equation(a2, z0)
                fb = _odd_equation(min(b2, z0 - 1e-8), z0)
                if fa * fb < 0:
                    root = optimize.brentq(lambda z: _odd_equation(z, z0), a2, min(b2, z0 - 1e-8), maxiter=200)
                    if 0 < root < z0:
                        roots.append(root)
            except Exception:
                pass

    # Перетворюємо з параметру z в енергію: k = 2z / L ; E = (hbar^2 k^2) / (2m)
    roots = sorted(set([float(r) for r in roots]))
    energies = []
    for z in roots:
        k = 2.0 * z / L
        E = (HBAR**2 * k**2) / (2.0 * m)
        # Перевіряємо, що E < U0
        if E < U0 - 1e-12:
            energies.append(E)
    return energies

# ---------------------------
# Barrier / Step solver - більш стабільна реалізація
# ---------------------------
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
        # Безпечні перетворення на скаляр
        k1 = complex(k1)

        if E > U0:
            k2 = get_k(E, self.m, U0)
            k2 = complex(k2)
            # Амплітуди відбиття і пропускання (скалярні)
            R_amp = (k1 - k2) / (k1 + k2)
            T_amp = 2.0 * k1 / (k1 + k2)

            psi = np.zeros_like(x, dtype=complex)
            left_mask = x < 0
            right_mask = x >= 0

            psi[left_mask] = np.exp(1j * k1 * x[left_mask]) + R_amp * np.exp(-1j * k1 * x[left_mask])
            psi[right_mask] = T_amp * np.exp(1j * k2 * x[right_mask])

            # Коефіцієнти потоків
            k1_r = k1.real if abs(k1.real) > 1e-18 else 1e-18
            k2_r = k2.real if abs(k2.real) > 1e-18 else 1e-18
            T = (k2_r / k1_r) * (abs(T_amp)**2)
            R = abs(R_amp)**2
            return np.real(psi), np.abs(psi)**2, T, R
        else:
            # E < U0: відбиття з експоненційним спадом справа
            k2 = get_k(E, self.m, U0)
            # k2 буде чисто уявним -> беремо kappa = imag(k2)
            kappa = abs(complex(k2).imag)
            psi = np.zeros_like(x, dtype=complex)
            left_mask = x < 0
            right_mask = x >= 0
            # Стояча хвиля ліворуч (інтерференція падаючої та відбитої)
            # Нормалізація амплітуди можна тут взяти =1 для візуалізації
            psi[left_mask] = np.exp(1j * k1 * x[left_mask]) + np.exp(-1j * k1 * x[left_mask])
            # Експонента справа
            psi[right_mask] = np.exp(-kappa * x[right_mask])
            # Для таких випадків T = 0, R = 1
            return np.real(psi), np.abs(psi)**2, 0.0, 1.0

    def solve_rectangular(self, E, U0, L, x):
        """
        Розв'язок для прямокутного бар'єра шириною L.
        Повертає psi_real, prob_density, T, R
        Виконує захисти від overflow у sinh/cosh при великих аргументах.
        """
        x = np.array(x, dtype=float)
        k1 = complex(get_k(E, self.m, 0.0))
        k2_complex = complex(get_k(E, self.m, U0))

        # Обчислення T стабільно
        T = 0.0
        R = 1.0
        try:
            if E > U0:
                k2r = k2_complex.real
                # Формула через інтерференцію в бар'єрі
                # Обмежуємо sin аргумент, використовуємо numpy для комплексного sin
                denom = 1.0 + (U0**2 * (np.sin(k2r * L)**2)) / (4.0 * E * (E - U0))
                if denom == 0:
                    T = 0.0
                else:
                    T = 1.0 / denom
            else:
                # E < U0
                kappa = abs(k2_complex.imag)
                # Якщо kappa*L дуже велике -> тунелювання експоненційно мале
                if kappa * L > 100.0:
                    T = 0.0
                else:
                    denom = 1.0 + (U0**2 * (np.sinh(kappa * L)**2)) / (4.0 * E * (U0 - E))
                    T = 1.0 / denom
            R = max(0.0, 1.0 - T)
        except Exception:
            T = 0.0
            R = 1.0

        # Тепер зшиваємо повний хвильовий розв'язок (захищено від overflow)
        # Для побудови psi(x) використовуємо аналітичні амплітуди
        psi = np.zeros_like(x, dtype=complex)
        left_mask = x < 0
        mid_mask = (x >= 0) & (x <= L)
        right_mask = x > L

        # Обчислення амплітуд (використаємо стабільну формулу)
        # Щоб уникнути ділення на нуль чи надвеликих значень, перевіряємо denom_t
        try:
            # Відповідно до теорії: t_amp та r_amp можна отримати через матричний метод.
            # Для простоти та стабільності скористаємося аналітичними виразами (але в комплексній формі):
            # Тут робимо безпечний розрахунок через формулу багатошарових шарів
            k1c = k1
            k2c = k2_complex
            # Щоб уникнути переповнення при exp(complex * large), обмежимо модулі експонент на етапі побудови psi
            # Але спочатку розрахуємо амплітуди:
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

        # Побудова хвилі з контрольованими експонентами (замінюємо надто великі аргументи)
        def safe_exp(z):
            # якщо абсолютне значення уявної/дійсної частини надто велике, обрізаємо за модулем
            # але для фізичної візуалізації зазвичай x * imag(k) не буде надто великим тут
            return np.exp(z)

        # Ліва частина
        if np.any(left_mask):
            psi[left_mask] = np.exp(1j * k1 * x[left_mask]) + r_amp * np.exp(-1j * k1 * x[left_mask])

        # Середина бар'єру
        if np.any(mid_mask):
            # Обчислимо коефіцієнти A,B через зшивку на x=0 та x=L
            # Замінимо точну систему на простішу, що працює стабільніше: використаємо лінійну алгебру
            # Побудуємо матрицю для умов неперервності psi та derivative psi' у x=0 та x=L
            try:
                # Значення в x=0: left at 0- і mid at 0+
                x0 = 0.0
                xL = L
                # Матриця з умов
                # psi_mid(x) = A e^{i k2 x} + B e^{-i k2 x}
                # derivative: i k2 A e^{i k2 x} - i k2 B e^{-i k2 x}
                M = np.array([
                    [np.exp(1j * k2c * x0), np.exp(-1j * k2c * x0)],
                    [1j * k2c * np.exp(1j * k2c * x0), -1j * k2c * np.exp(-1j * k2c * x0)]
                ], dtype=complex)
                # rhs: значення зліва в x=0: psi_left(0) та derivative
                psi_left_0 = 1.0 + r_amp  # left amplitude at 0
                psi_left_der_0 = 1j * k1 * (1.0 - r_amp)
                b = np.array([psi_left_0, psi_left_der_0], dtype=complex)
                sol = linalg.solve(M, b)
                Acoef, Bcoef = sol[0], sol[1]
            except Exception:
                Acoef, Bcoef = 0.0, 0.0

            # Тепер заповнимо всередині
            psi[mid_mask] = Acoef * np.exp(1j * k2c * x[mid_mask]) + Bcoef * np.exp(-1j * k2c * x[mid_mask])

        # Права частина
        if np.any(right_mask):
            psi[right_mask] = t_amp * np.exp(1j * k1 * x[right_mask])

        # Повертаємо реальну частину для малювання та щільність ймовірності
        psi_real = np.real(psi)
        prob = np.abs(psi)**2
        return psi_real, prob, T, R

# ---------------------------
# Time-dependent solver (Crank-Nicolson) - базова реалізація
# ---------------------------
class TimeDependentSolver:
    """Простий Crank-Nicolson для 1D TDSE в нерівномірній (але рівномірній тут) сітці."""

    def __init__(self, m, Nx=800, L_space=2e-8):
        self.m = float(m)
        self.Nx = int(Nx)
        self.L_space = float(L_space)
        self.dx = L_space / (self.Nx - 1)
        self.x = np.linspace(-L_space/2, L_space/2, self.Nx)
        self.alpha = 1j * HBAR / (2.0 * self.m * (self.dx**2))

    def construct_matrices(self, V, dt):
        """
        Повертає матриці A, B для розв'язування A psi_{n+1} = B psi_n
        """
        N = self.Nx
        r = 1j * HBAR * dt / (2.0 * (self.dx**2) * self.m)
        # Тридіагальна матриця
        main_diag_A = np.ones(N, dtype=complex) + 1j * dt * (V / (2.0 * HBAR)) + 2.0 * r
        off_diag = -r * np.ones(N-1, dtype=complex)
        A = diags([off_diag, main_diag_A, off_diag], offsets=[-1, 0, 1], format='csc')
        main_diag_B = np.ones(N, dtype=complex) + -1j * dt * (V / (2.0 * HBAR)) - 2.0 * r
        B = diags([+r * np.ones(N-1, dtype=complex), main_diag_B, +r * np.ones(N-1, dtype=complex)], offsets=[-1, 0, 1], format='csc')
        return A, B

    def init_gaussian_packet(self, x0= -3e-9, sigma=5e-10, k0=5e9):
        """
        Ініціалізація гаусовського пакета: psi ~ exp(-(x-x0)^2/(4sigma^2) + i k0 x)
        Нормуємо хвильову функцію.
        """
        psi = np.exp(- (self.x - x0)**2 / (4.0 * sigma**2) + 1j * k0 * self.x)
        # Нормалізація
        norm = np.sqrt(np.trapz(np.abs(psi)**2, self.x))
        psi = psi / norm
        return psi

    def simulate_packet(self, Ekin, U0_barrier, dt=1e-18, steps=150, barrier_center=0.0, barrier_width=2e-9):
        """
        Проста симуляція пакета зі статичним бар'єром.
        Повертає x, psi_final, матриці A,B та потенціал V (щоб можна було візуалізувати).
        """
        # Побудова потенціалу
        V = np.zeros_like(self.x)
        # Розташування бар'єру у середині сітки
        mask = (self.x >= barrier_center - barrier_width/2) & (self.x <= barrier_center + barrier_width/2)
        V[mask] = U0_barrier

        # Початковий пакет: виберемо k0 з Ekin = (hbar k0)^2 / (2m) => k0 = sqrt(2mE)/hbar
        if Ekin <= 0:
            k0 = 0.0
        else:
            k0 = np.sqrt(2.0 * self.m * Ekin) / HBAR

        psi = self.init_gaussian_packet(x0 = -self.L_space * 0.35, sigma = self.L_space * 0.03, k0 = k0)

        # Побудова матриць
        A, B = self.construct_matrices(V, dt)

        # Повертаємо початкові дані та матриці для інтегрування зверху
        return self.x, psi, A, B, V

# -------------------------------------------------------------------------
# 3. ФУНКЦІЇ ВІЗУАЛІЗАЦІЇ (HELPER)
# -------------------------------------------------------------------------
def draw_arrow(ax, x1, x2, y, text, color='white'):
    """Малює стрілку розміру."""
    ax.annotate('', xy=(x1, y), xytext=(x2, y), arrowprops=dict(arrowstyle='<->', color=color))
    ax.text((x1 + x2) / 2.0, y, text, ha='center', va='bottom', color=color,
            bbox=dict(facecolor='#0e1117', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.1'))

def plot_setup(ax, title, U_max):
    """Базове налаштування графіків (колір, осі, назва)."""
    ax.set_title(title, color='white')
    ax.set_xlabel("x (м)", color='white')
    ax.set_ylabel("Енергія / Ψ", color='white')
    # Коректний діапазон осі y
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
# 4. ГОЛОВНА ЛОГІКА ДОДАТКУ (MAIN)
# -------------------------------------------------------------------------
def main():
    # Ініціалізуємо session_state ключі, щоб кнопки працювали коректно
    if 'run_calc' not in st.session_state:
        st.session_state['run_calc'] = False

    st.sidebar.title("🎛 Панель Керування")

    # Вибір системи (записуємо в sys_type)
    sys_type = st.sidebar.selectbox("1. Система:",
                                    ["Потенціальна Яма", "Потенціальний Бар'єр", "Гармонічний Осцилятор", "🌊 Хвильовий Пакет"])

    sub_type = None
    if sys_type == "Потенціальна Яма":
        sub_type = st.sidebar.radio("Тип стінок:", ["Нескінченні стінки", "Кінцеві стінки"])
    elif sys_type == "Потенціальний Бар'єр":
        sub_type = st.sidebar.radio("Тип:", ["Сходинка", "Прямокутний бар'єр"])
    elif sys_type == "Гармонічний Осцилятор":
        sub_type = "Стандарт"
    else:
        sub_type = None

    st.sidebar.markdown("---")
    st.sidebar.header("2. Параметри")

    params = {}

    # Частинка
    particle_name = st.sidebar.selectbox("Частинка:", ["Електрон", "Протон", "Мюон"])
    # Більш реалістичне значення массы мюона ~206.768*m_e
    mass_map = {"Електрон": M_E, "Протон": M_P, "Мюон": M_E * 206.768}
    params['m'] = float(mass_map[particle_name])

    # Для більшості режимів потрібна ширина L
    if sys_type != "Гармонічний Осцилятор":
        params['L'] = st.sidebar.number_input("Ширина L (м)", value=1e-20, step=1e-10, format="%.2e")

    # Потенціал U0, енергія E
    if sys_type in ["Потенціальний Бар'єр", "🌊 Хвильовий Пакет"] or (sys_type == "Потенціальна Яма" and sub_type == "Кінцеві стінки"):
        params['U0'] = st.sidebar.number_input("Потенціал U₀ (Дж)", value=50.0 * EV, step=1.6e-20, format="%.2e")

    if sys_type in ["Потенціальний Бар'єр", "🌊 Хвильовий Пакет"]:
        params['E'] = st.sidebar.number_input("Енергія E (Дж)", value=5.0 * EV, step=1.6e-20, format="%.2e")

    if sys_type == "Гармонічний Осцилятор":
        # Додаємо кнопку +/- через step
        params['omega'] = st.sidebar.number_input("Частота ω (рад/с)", value=5e15, format="%.2e", step=1e13)

    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 РОЗРАХУВАТИ"):
        st.session_state['run_calc'] = True

    # Головна панель
    if st.session_state.get('run_calc', False):
        st.title(f"Результати: {sys_type} ({sub_type})")
        m = params.get('m', M_E)
        L = params.get('L', 1e-9)
        U0 = params.get('U0', 0.0)
        E = params.get('E', 0.0)
        omega = params.get('omega', 1e15)

        # ------------------------------------------------------------------
        # 1. НЕСКІНЧЕННА ЯМА
        # ------------------------------------------------------------------
        if sys_type == "Потенціальна Яма" and sub_type == "Нескінченні стінки":
            energies = solve_inf_well(L, m, 10)
            n_viz = st.slider("Рівень n", 1, min(10, len(energies)), 1, key='inf_n_slider')
            E_n = energies[n_viz - 1]

            st.success(f"E = {E_n / EV:.6f} еВ")
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_setup(ax, f"Нескінченна Яма (n={n_viz})", max(E_n, U0, 1e-20))

            ax.vlines([0, L], -0.05 * E_n, energies[-1] * 1.1, colors='white', linewidth=3)
            ax.hlines(0, -L * 0.1, L * 1.1, color='white')
            ax.hlines(E_n, -L * 0.1, L * 1.1, color='red', linestyle='--', label=f'$E_{n_viz}$')

            x = np.linspace(0, L, 1000)
            psi = psi_inf_well(x, L, n_viz)
            # масштабування для накладання на енергійну шкалу
            scale = E_n * 0.4
            if np.max(np.abs(psi)) > 0:
                psi_plot = E_n + psi / np.max(np.abs(psi)) * scale
            else:
                psi_plot = E_n + psi * scale

            ax.plot(x, psi_plot, label=r'Re($\Psi$)', color='cyan', lw=2)
            ax.fill_between(x, E_n, psi_plot, color='cyan', alpha=0.1)

            draw_arrow(ax, 0, L, -E_n * 0.05, f"L={L:.1e} м")
            ax.legend(loc='upper right')
            st.pyplot(fig)

        # ------------------------------------------------------------------
        # 2. КІНЦЕВА ЯМА
        # ------------------------------------------------------------------
        elif sys_type == "Потенціальна Яма" and sub_type == "Кінцеві стінки":
            N, z0 = finite_well_solver(m, L, U0)
            st.success(f"Орієнтовна кількість рівнів: {N} (z0={z0:.3f})")

            energies_found = solve_finite_well(m, L, U0)
            if len(energies_found) == 0:
                st.info("Зв'язаних рівнів не знайдено або вони занадто близькі до верхньої межі U0.")
                E_n = None
            else:
                limit_N = min(len(energies_found), 6)
                n_viz = st.slider("Рівень n", 1, limit_N, 1, key='fin_n_slider')
                E_n = energies_found[n_viz - 1]
                st.info(f"E_{n_viz} = {E_n / EV:.6f} еВ (знайдено чисельно, U0={U0/EV:.2f} еВ)")

            fig, ax = plt.subplots(figsize=(10, 6))
            plot_setup(ax, "Кінцева Яма", max(U0, max(energies_found) if energies_found else U0, 1e-20))

            x = np.linspace(-L, 2.0 * L, 800)
            U_pot = np.zeros_like(x)
            U_pot[(x < 0) | (x > L)] = U0
            ax.plot(x, U_pot, 'w-', lw=2, label='U(x)')

            draw_arrow(ax, 0, L, U0 * 0.05, "L")

            if E_n is not None:
                ax.hlines(E_n, -L * 0.5, L * 1.5, colors='r', linestyles='--')
                # Для наочності можемо намалювати наближений вигляд хвильової функції всередині
                x_in = np.linspace(0, L, 400)
                # Візьмемо аналітичну форму для нескінченної ями як грубу апроксимацію форм-фактора
                psi_in = psi_inf_well(x_in, L, n_viz)
                scale = E_n * 0.3
                psi_plot = E_n + psi_in / np.max(np.abs(psi_in)) * scale
                ax.plot(x_in, psi_plot, color='cyan', label='ψ (орієнтовно)')
                ax.fill_between(x_in, E_n, psi_plot, color='cyan', alpha=0.1)
            else:
                st.info("Хвильова функція опущена (немає зв'язаних рівнів).")
            ax.legend(loc='upper right')
            st.pyplot(fig)

        # ------------------------------------------------------------------
        # 3. ГАРМОНІЧНИЙ ОСЦИЛЯТОР
        # ------------------------------------------------------------------
        elif sys_type == "Гармонічний Осцилятор":
            # створюємо список енергій
            energies = [calc_harmonic_energy(omega, n) for n in range(10)]
            n_viz = st.slider("Рівень n", 0, 9, 0, key='osc_n_slider')
            E_n = energies[n_viz]

            st.success(f"E_{n_viz} = {E_n:.4e} Дж ({E_n / EV:.6f} еВ)")

            fig, ax = plt.subplots(figsize=(10, 6))
            # класична поворотна точка
            x_turn = np.sqrt(2.0 * E_n / (m * omega**2)) if (m > 0 and omega > 0 and E_n > 0) else 1e-9
            x_turn_max = np.sqrt(2.0 * energies[-1] / (m * omega**2)) if (m > 0 and omega > 0 and energies[-1] > 0) else x_turn
            x_lim = max(x_turn_max * 1.2, 1e-10)
            x = np.linspace(-x_lim, x_lim, 800)
            U = 0.5 * m * omega**2 * x**2

            plot_setup(ax, "Гармонічний Осцилятор", max(energies[-1], U.max(), 1e-20))
            ax.plot(x, U, 'w-', label='U(x)')
            ax.hlines(E_n, -x_lim, x_lim, colors='r', linestyles='--')

            psi = psi_oscillator(x, m, omega, n_viz)
            # scale для накладення
            if np.max(np.abs(psi)) > 0:
                psi_plot = E_n + psi / np.max(np.abs(psi)) * (energies[1] - energies[0]) * 0.8
            else:
                psi_plot = E_n + psi * (energies[1] - energies[0]) * 0.8
            prob_plot = E_n + (psi**2) / np.max(psi**2 + 1e-30) * (energies[1] - energies[0]) * 0.8

            ax.plot(x, psi_plot, label=r'$\Psi$', color='cyan')
            ax.plot(x, prob_plot, label=r'$|\Psi|^2$', color='magenta', linestyle=':')
            draw_arrow(ax, -x_turn, x_turn, E_n * 1.05, f"2A={2.0 * x_turn:.1e} м")
            ax.legend(loc='upper right')
            st.pyplot(fig)

        # ------------------------------------------------------------------
        # 4. СХОДИНКА
        # ------------------------------------------------------------------
        elif sys_type == "Потенціальний Бар'єр" and sub_type == "Сходинка":
            m_val = m
            E_val = E
            U0_val = U0
            x_viz = np.linspace(-2e-9, 2e-9, 1000)

            solver = BarrierSolver(m_val)
            psi_real, psi_prob, T, R = solver.solve_step(E_val, U0_val, x_viz)

            # Виводимо метрики
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("T", f"{T:.6f}")
                st.metric("R", f"{R:.6f}")

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
        # 5. ПРЯМОКУТНИЙ БАР'ЄР
        # ------------------------------------------------------------------
        elif sys_type == "Потенціальний Бар'єр" and sub_type == "Прямокутний бар'єр":
            m_val = m
            E_val = E
            U0_val = U0
            L_val = L

            solver = BarrierSolver(m_val)
            x = np.linspace(-2.0 * L_val, 3.0 * L_val, 1200)
            psi_real, psi_prob, T, R = solver.solve_rectangular(E_val, U0_val, L_val, x)

            # Метрики
            st.metric("T", f"{T:.6e}")
            st.metric("R", f"{R:.6f}")

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

        # ------------------------------------------------------------------
        # 6. ХВИЛЬОВИЙ ПАКЕТ (TDSE)
        # ------------------------------------------------------------------
        elif sys_type == "🌊 Хвильовий Пакет":
            st.warning("TDSE: чисельні методи. Може зайняти трохи часу в залежності від Nx/кроків.")
            L_space = 2e-8
            U0_bar = params.get('U0', 50.0 * EV)
            E_kin = params.get('E', 5.0 * EV)

            col_run, col_opts = st.columns([1, 2])
            with col_opts:
                steps = st.number_input("Кроків (макс графіки)", min_value=10, max_value=2000, value=150)
                dt = st.number_input("Δt (с)", value=1e-18, format="%.1e")
                Nx = st.number_input("Nx (сітка)", min_value=200, max_value=3000, value=800)

            solver = TimeDependentSolver(params['m'], Nx=int(Nx), L_space=L_space)
            x_grid, psi, A, B, V = solver.simulate_packet(E_kin, U0_bar, dt=dt, steps=int(steps))

            # Кнопки запуску анімації/окремі кадри
            if st.button("▶️ Запустити Анімацію"):
                plot_holder = st.empty()
                # Лічильник для малювання
                psi_current = psi.copy()
                A_csc = A  # sparse
                B_csc = B

                # Використовуємо sparse-розв'язувач (spsolve) для прискорення
                from scipy.sparse.linalg import splu
                try:
                    lu = splu(A_csc.tocsc())
                except Exception:
                    lu = None

                for i in range(int(steps)):
                    # Обчислюємо rhs = B * psi_current
                    rhs = B_csc.dot(psi_current)
                    if lu is not None:
                        psi_current = lu.solve(rhs)
                    else:
                        psi_current = linalg.spsolve(A_csc, rhs)

                    if i % max(1, int(max(1, steps // 80))) == 0:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        plot_setup(ax, f"t крок = {i}", max(np.max(V), np.max(np.abs(psi_current)**2)) + 1e-20)
                        ax.plot(solver.x * 1e9, V / (np.max(V) + 1e-30) * np.max(np.abs(psi_current)**2), color='gray', alpha=0.5, label="Бар'єр (масштабовано)")
                        ax.plot(solver.x * 1e9, np.abs(psi_current)**2, color='cyan', lw=2, label=r'$|\Psi(t)|^2$')
                        ax.set_xlabel("x (нм)")
                        ax.set_title(f"t = {i * dt:.2e} с (крок {i})")
                        ax.legend(loc='upper right')
                        plot_holder.pyplot(fig)
                        plt.close(fig)

                st.success("Анімація завершена.")
            else:
                st.info("Натисніть '▶️ Запустити Анімацію' для запуску TDSE симуляції.")

        else:
            st.info("Оберіть параметри для конкретного режиму і натисніть '🚀 РОЗРАХУВАТИ'.")

    else:
        st.title("Квантовий Симулятор Ultimate")
        st.markdown("Налаштуйте параметри зліва та натисніть **🚀 РОЗРАХУВАТИ**.")

if __name__ == "__main__":
    main()