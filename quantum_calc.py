import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go 
from scipy import constants
from scipy.special import hermite

# --- 1. КОНСТАНТИ ---
HBAR = constants.hbar
M_E = constants.m_e
EV = constants.electron_volt

# --- 2. МАТЕМАТИЧНИЙ ДВИГУН (ЯМИ ТА БАР'ЄРИ) ---

def calc_infinite_well_energy(m, L, n):
    """Енергія в нескінченній ямі"""
    return (n**2 * np.pi**2 * HBAR**2) / (2 * m * L**2)

def calc_harmonic_energy(omega, n):
    """Енергія гармонічного осцилятора"""
    return HBAR * omega * (n + 0.5)

def calc_step_coefficients(m, E, U0):
    """Розрахунок для сходинки"""
    if E < 0 or m <= 0: return "Error", 0, 0, 0, 0
    k1 = np.sqrt(2 * m * E) / HBAR
    
    if E > U0:
        k2 = np.sqrt(2 * m * (E - U0)) / HBAR
        if (k1 + k2) == 0: return "Pass", 0, 0, k1, k2 
        R = ((k1 - k2) / (k1 + k2))**2
        T = 1 - R
        return "Pass", R, T, k1, k2
    else:
        kappa = np.sqrt(2 * m * (U0 - E)) / HBAR
        depth = 1 / kappa if kappa > 0 else 0
        return "Reflect", 1.0, 0.0, k1, kappa

def calc_barrier_tunneling(m, E, U0, L):
    """Розрахунок для бар'єра"""
    if E >= U0:
        k2 = np.sqrt(2 * m * (E - U0)) / HBAR
        with np.errstate(divide='ignore', invalid='ignore'):
            if E == U0: T = 1.0 
            else:
                sin_term = np.sin(k2 * L)**2
                denom = 1 + (U0**2 * sin_term) / (4 * E * (E - U0))
                T = 1 / denom if denom != 0 else 0
    else:
        kappa = np.sqrt(2 * m * (U0 - E)) / HBAR
        with np.errstate(divide='ignore', invalid='ignore'):
            sinh_term = np.sinh(kappa * L)**2
            denom_val = 4 * E * (U0 - E)
            if denom_val == 0: T = 0 
            else:
                denom = 1 + (U0**2 * sinh_term) / denom_val
                T = 1 / denom
    R = 1 - T
    return T, R

def finite_well_solver(m, L, U0):
    """Розрахунок рівнів у кінцевій ямі"""
    if U0 <= 0: return 0, 0
    z0 = (L / 2) * np.sqrt(2 * m * U0) / HBAR
    N = 1 + int((2 * z0) / np.pi)
    return N, z0

# --- 3. ФУНКЦІЇ ВІЗУАЛІЗАЦІЇ (HELPER) ---

def draw_arrow(ax, x1, x2, y, text, color='black'):
    ax.annotate('', xy=(x1, y), xytext=(x2, y), arrowprops=dict(arrowstyle='<->', color=color))
    ax.text((x1+x2)/2, y, text, ha='center', va='bottom', color=color, 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# --- 4. ОКРЕМІ МОДУЛІ ---

def run_spin_visualization():
    st.header("🌀 Сфера Блоха (Спін 1/2)")
    st.info("Візуалізація квантового стану кубіта (спіна) як вектору на сфері.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        theta = st.slider("Кут θ (0...π)", 0.0, np.pi, 0.0, 0.01)
        phi = st.slider("Кут φ (0...2π)", 0.0, 2*np.pi, 0.0, 0.01)
        
        p_up = np.cos(theta/2)**2
        p_down = np.sin(theta/2)**2
        
        st.write(f"**Ймовірності:**")
        st.write(f"↑ (Вгору): {p_up:.2%}")
        st.write(f"↓ (Вниз): {p_down:.2%}")
        
        st.latex(r"|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle")
        
    with col2:
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        xs = np.cos(u)*np.sin(v)
        ys = np.sin(u)*np.sin(v)
        zs = np.cos(v)
        
        fig = go.Figure()
        fig.add_trace(go.Surface(x=xs, y=ys, z=zs, opacity=0.1, showscale=False, colorscale='Blues'))
        fig.add_trace(go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], mode='lines+markers', 
                                   line=dict(color='red', width=10), marker=dict(size=5)))
        fig.add_trace(go.Scatter3d(x=[0,0,1.2,0,0], y=[0,0,0,1.2,0], z=[1.2,-1.2,0,0,0], 
                                   mode='text', text=['|0⟩', '|1⟩', '+X', '+Y', ''], showlegend=False))
        
        fig.update_layout(width=600, height=500, margin=dict(l=0, r=0, b=0, t=0), showlegend=False)
        st.plotly_chart(fig)


def run_entanglement_simulation():
    st.header("🔗 Квантова Заплутаність (Експеримент Белла)")
    
    # Вибір типу симуляції всередині розділу
    sim_type = st.radio("Оберіть демонстрацію:", 
        ["1. Візуалізація Штерна-Герлаха (Схема)", 
         "2. Експеримент Белла (Графік кореляції)"])

    st.divider()

    # === 1. СХЕМА ШТЕРНА-ГЕРЛАХА (Згідно з вашим описом) ===
    if sim_type == "1. Візуалізація Штерна-Герлаха (Схема)":
        st.subheader("Експеримент з парою заплутаних електронів")
        st.markdown("""
        **Опис:**
        * Джерело випускає пару електронів із сумарним спіном 0.
        * Вони розлітаються до магнітів Штерна-Герлаха.
        * Якщо магніти орієнтовані однаково, один електрон летить до **N**, інший — до **S**.
        * Це завжди **протилежні** напрямки (антикореляція).
        """)
        
        if st.button("🔴 Запустити пару електронів"):
            # Випадково обираємо: (Вгору, Вниз) або (Вниз, Вгору)
            # 0 = Вгору (до S магніту, відхилення до N полюса на екрані)
            # 1 = Вниз (до N магніту, відхилення до S полюса на екрані)
            # На вашій картинці: "deflection toward north pole" = вгору на схемі? 
            # Зазвичай електрони відхиляються силою Лоренца або градієнтом поля.
            # Будемо вважати як на картинці: один вгору, інший вниз.
            
            outcome = np.random.choice(['up_down', 'down_up'])
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-2, 2)
            ax.axis('off')
            
            # Джерело
            circle = plt.Circle((0, 0), 0.2, color='purple', label='Джерело')
            ax.add_patch(circle)
            ax.text(0, -0.5, "Джерело (S=0)", ha='center')
            
            # Магніти (Лівий)
            ax.add_patch(plt.Rectangle((-3.5, 0.5), 1, 1.5, fc='white', ec='black')) # Верхній (S)
            ax.text(-3, 1.25, "S", ha='center', fontsize=12)
            # Трикутник знизу (N)
            triangle_left = plt.Polygon([(-3.5, -1.5), (-2.5, -1.5), (-3, -0.5)], fc='white', ec='black')
            ax.add_patch(triangle_left)
            ax.text(-3, -1.2, "N", ha='center', fontsize=12)

            # Магніти (Правий)
            ax.add_patch(plt.Rectangle((2.5, 0.5), 1, 1.5, fc='white', ec='black')) # Верхній (S)
            ax.text(3, 1.25, "S", ha='center', fontsize=12)
            # Трикутник знизу (N)
            triangle_right = plt.Polygon([(2.5, -1.5), (3.5, -1.5), (3, -0.5)], fc='white', ec='black')
            ax.add_patch(triangle_right)
            ax.text(3, -1.2, "N", ha='center', fontsize=12)

            # Траєкторії
            x_left = np.linspace(-3, -0.2, 50)
            x_right = np.linspace(0.2, 3, 50)
            
            # Випадок 1: Лівий -> N (Вниз), Правий -> S (Вгору) - як на рис. 2
            # Але на рис. 2 стрілка зліва йде ВНИЗ (до N трикутника?), справа ВГОРУ (до S прямокутника?)
            # Давайте відтворимо точно як на картинці.
            
            if outcome == 'down_up':
                # Лівий електрон: відхиляється ВНИЗ (до N)
                y_left = -0.5 * (np.exp(0.5 * (-x_left - 0.2)) - 1)
                ax.arrow(x_left[-1], y_left[-1], x_left[0]-x_left[-1], y_left[0]-y_left[-1], 
                         head_width=0.1, fc='black', length_includes_head=True)
                
                # Правий електрон: відхиляється ВГОРУ (до S)
                y_right = 0.5 * (np.exp(0.5 * (x_right - 0.2)) - 1)
                ax.arrow(x_right[0], y_right[0], x_right[-1]-x_right[0], y_right[-1]-y_right[0], 
                         head_width=0.1, fc='black', length_includes_head=True)
                
                res_text = "Лівий -> Південь (N), Правий -> Північ (S)"
                
            else: # up_down
                # Навпаки
                y_left = 0.5 * (np.exp(0.5 * (-x_left - 0.2)) - 1) # Вгору
                ax.arrow(x_left[-1], y_left[-1], x_left[0]-x_left[-1], y_left[0]-y_left[-1], 
                         head_width=0.1, fc='black', length_includes_head=True)

                y_right = -0.5 * (np.exp(0.5 * (x_right - 0.2)) - 1) # Вниз
                ax.arrow(x_right[0], y_right[0], x_right[-1]-x_right[0], y_right[-1]-y_right[0], 
                         head_width=0.1, fc='black', length_includes_head=True)
                
                res_text = "Лівий -> Північ (S), Правий -> Південь (N)"
                
            st.pyplot(fig)
            st.success(f"Результат вимірювання: **{res_text}**")
            st.info("Як бачимо, результати завжди протилежні, хоча кожен окремо - випадковий.")

    # === 2. ЕКСПЕРИМЕНТ БЕЛЛА (З ГРАФІКОМ) ===
    elif sim_type == "2. Експеримент Белла (Графік кореляції)":
        st.markdown(r"""
        **Перевірка нерівності Белла:**
        1. Кожен спостерігач має детектор, який можна повернути під кутом $\alpha$ і $\beta$.
        2. Вони отримують результат: $+1$ (↑) або $-1$ (↓).
        3. Квантова механіка каже: якщо кути збігаються ($\alpha = \beta$), результати **ЗАВЖДИ** протилежні.
        4. Завдання: перевірити залежність кореляції від різниці кутів $\theta = |\alpha - \beta|$.
        """)
        
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("🛠 Налаштування")
            angle_a = st.slider("Кут Аліси (α)", 0, 360, 0, step=15)
            angle_b = st.slider("Кут Боба (β)", 0, 360, 45, step=15)
            theta_deg = abs(angle_a - angle_b)
            
            st.info(f"Різниця кутів: **{theta_deg}°**")
            
            # Теоретична кореляція E = -cos(theta)
            corr_theory = -np.cos(np.radians(theta_deg))
            st.metric("Квантова кореляція (Теорія)", f"{corr_theory:.4f}")
            
        with c2:
            st.subheader("🎲 Симуляція (Монте-Карло)")
            n_shots = st.select_slider("Кількість вимірювань", [100, 1000, 5000, 10000], value=1000)
            
            if st.button("Запустити експеримент"):
                theta_rad = np.radians(theta_deg)
                # P_diff = cos^2(theta/2), P_same = sin^2(theta/2)
                prob_diff = np.cos(theta_rad/2)**2
                
                # Симуляція
                random_vals = np.random.random(n_shots)
                diff_count = np.sum(random_vals < prob_diff)
                same_count = n_shots - diff_count
                
                # Кореляція E = (same - diff) / total
                # Оскільки diff це (-1)*(+1)=-1, а same це (+1)*(+1)=+1
                # E = (same - diff) / total
                corr_exp = (same_count - diff_count) / n_shots
                
                st.success(f"Результати: Різні={diff_count}, Однакові={same_count}")
                
                # Графік
                fig, ax = plt.subplots(figsize=(6, 3))
                angles = np.linspace(0, 360, 100)
                # Теорія: -cos(theta)
                ax.plot(angles, -np.cos(np.radians(angles)), 'k--', label=r'Теорія ($-\cos \theta$)')
                ax.plot([theta_deg], [corr_exp], 'ro', markersize=10, label='Ваш результат')
                
                ax.set_xlabel(r"Різниця кутів $\theta$ (градуси)")
                ax.set_ylabel("Кореляція")
                ax.axhline(0, color='gray', lw=0.5)
                ax.legend()
                st.pyplot(fig)


# --- 5. ГОЛОВНИЙ ІНТЕРФЕЙС ---

def main():
    st.set_page_config(layout="wide", page_title="Quantum Physics Solver")
    
    if 'calc_active' not in st.session_state:
        st.session_state['calc_active'] = False

    st.title("⚛️ Квантовий Розв'язувач")

    # --- САЙДБАР (ГОЛОВНЕ МЕНЮ) ---
    st.sidebar.header("1. Головне Меню")
    
    main_mode = st.sidebar.radio("Оберіть режим роботи:", 
                                 ["Задачі (Ями та Бар'єри)", 
                                  "🌀 Спін (Сфера Блоха)", 
                                  "🔗 Квантова Заплутаність"]) 
    
    # ================== РОЗДІЛ ЗАПЛУТАНОСТІ ==================
    if main_mode == "🔗 Квантова Заплутаність":
        run_entanglement_simulation()
        return 

    # ================== РОЗДІЛ СПІНА ==================
    if main_mode == "🌀 Спін (Сфера Блоха)":
        run_spin_visualization()
        return 
        
    # ================== РОЗДІЛ ЗАДАЧ ==================
    st.sidebar.markdown("---")
    st.sidebar.header("2. Налаштування Задачі")
    sys_type = st.sidebar.selectbox("Система:", ["Потенціальна Яма", "Потенціальний Бар'єр", "Гармонічний Осцилятор"])

    sub_type = None
    if sys_type == "Потенціальна Яма":
        sub_type = st.sidebar.radio("Тип стінок:", ["Нескінченні", "Кінцеві"])
    elif sys_type == "Потенціальний Бар'єр":
        sub_type = st.sidebar.radio("Тип:", ["Сходинка", "Прямокутний"])
    elif sys_type == "Гармонічний Осцилятор":
        sub_type = "Стандарт"

    st.sidebar.markdown("---")
    st.sidebar.header("3. Параметри")
    
    params = {}
    particle_name = st.sidebar.selectbox("Тип частинки:", ["Електрон", "Мюон", "Протон"])
    mass_map = {"Електрон": 1, "Мюон": 207, "Протон": 1836}
    params['m'] = M_E * mass_map[particle_name]
    st.sidebar.caption(f"m = {params['m']:.2e} кг")

    if sys_type == "Потенціальна Яма":
        params['L'] = st.sidebar.number_input("Ширина ями L (м)", value=1e-9, format="%.2e")
        if sub_type == "Кінцеві":
            params['U0'] = st.sidebar.number_input("Глибина U₀ (Дж)", value=50*EV, format="%.2e")
    
    elif sys_type == "Потенціальний Бар'єр":
        params['U0'] = st.sidebar.number_input("Висота бар'єра U₀ (Дж)", value=5*EV, format="%.2e")
        params['E'] = st.sidebar.number_input("Енергія E (Дж)", value=2*EV, format="%.2e")
        if sub_type == "Прямокутний":
            params['L'] = st.sidebar.number_input("Ширина бар'єра L (м)", value=1e-10, format="%.2e")
            
    elif sys_type == "Гармонічний Осцилятор":
        params['omega'] = st.sidebar.number_input("Частота ω (рад/с)", value=1e15, format="%.2e")

    st.sidebar.markdown("---")
    
    if st.sidebar.button("🚀 Розрахувати"):
        st.session_state['calc_active'] = True
        if 'viz_n' not in st.session_state:
            st.session_state['viz_n'] = 1 

    # --- ВИВІД РЕЗУЛЬТАТІВ ЗАДАЧ ---
    
    if st.session_state['calc_active']:
        st.header(f"Результати: {sys_type}")
        
        # 1. НЕСКІНЧЕННА ЯМА
        if sys_type == "Потенціальна Яма" and sub_type == "Нескінченні":
            energies = [calc_infinite_well_energy(params['m'], params['L'], n) for n in range(1, 6)]
            
            n_viz = st.slider("Головне Квантове Число (n)", 1, 5, 1, key='slider_inf_well')
            E_cur = energies[n_viz-1]
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.success(f"n={n_viz}: E = {E_cur:.4e} Дж")
                st.info(f"E = {E_cur/EV:.4f} еВ")
            
            with c2:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.vlines([0, params['L']], 0, energies[-1]*1.2, color='black', linewidth=3)
                ax.hlines(0, 0, params['L'], color='black')
                ax.hlines(E_cur, 0, params['L'], color='gray', linestyle='--', label=f'$E_{n_viz}$')
                
                x = np.linspace(0, params['L'], 300)
                psi = np.sqrt(2/params['L']) * np.sin(n_viz * np.pi * x / params['L'])
                prob = psi**2
                scale = E_cur * 0.5 
                
                ax.plot(x, E_cur + (psi / np.max(np.abs(psi))) * scale, color='blue', label=r'Хвильова функція ($\Psi$)')
                ax.plot(x, E_cur + (prob / np.max(prob)) * scale, color='red', label=r'Густина ($|\Psi|^2$)')
                ax.fill_between(x, E_cur, E_cur + (prob / np.max(prob)) * scale, alpha=0.1, color='red')
                
                draw_arrow(ax, 0, params['L'], -E_cur*0.1, f"L={params['L']:.1e}")
                
                ax.set_ylabel("Енергія")
                ax.legend(loc='upper right')
                st.pyplot(fig)

        # 2. КІНЦЕВА ЯМА
        elif sys_type == "Потенціальна Яма" and sub_type == "Кінцеві":
            N, z0 = finite_well_solver(params['m'], params['L'], params['U0'])
            st.success(f"Кількість рівнів: {N} (Параметр z₀={z0:.2f})")
            
            n_viz = 1
            if N > 0:
                limit_N = min(N, 50)
                n_viz = st.slider(f"Рівень n (всього {N})", 1, limit_N, 1, key='slider_fin_well')
            else:
                n_viz = 0
                st.warning("Яма занадто мала")

            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.linspace(-params['L'], 2*params['L'], 400)
            U_pot = np.zeros_like(x)
            U_pot[(x < 0) | (x > params['L'])] = params['U0']
            
            ax.plot(x, U_pot, 'k-', linewidth=2, label='Потенціал U(x)')
            ax.fill_between(x, U_pot, alpha=0.1, color='gray')
            draw_arrow(ax, 0, params['L'], params['U0']*0.5, "L")
            
            if n_viz > 0:
                E_approx = calc_infinite_well_energy(params['m'], params['L'], n_viz)
                if E_approx >= params['U0']: E_approx = params['U0'] * 0.9
                
                ax.hlines(E_approx, 0, params['L'], color='gray', linestyle='--', label=f'$E_{n_viz}$')
                
                x_in = np.linspace(0, params['L'], 200)
                psi_in = np.sin(n_viz * np.pi * x_in / params['L'])
                prob_in = psi_in**2
                scale = params['U0'] * 0.2
                
                ax.plot(x_in, E_approx + psi_in * scale, color='blue', label=r'$\Psi$')
                ax.plot(x_in, E_approx + prob_in * scale, color='red', label=r'$|\Psi|^2$')
                ax.fill_between(x_in, E_approx, E_approx + prob_in * scale, alpha=0.1, color='red')

            ax.legend(loc='upper right')
            st.pyplot(fig)

        # 3. ОСЦИЛЯТОР
        elif sys_type == "Гармонічний Осцилятор":
            energies = [calc_harmonic_energy(params['omega'], n) for n in range(6)]
            
            n_viz = st.slider("Квантове число n", 0, 5, 0, key='slider_osc')
            E_n = energies[n_viz]
            
            st.success(f"E_{n_viz} = {E_n:.4e} Дж ({E_n/EV:.4f} еВ)")
                
            fig, ax = plt.subplots(figsize=(8, 6))
            
            if params['m'] > 0 and params['omega'] > 0:
                x_turn = np.sqrt(2 * energies[-1] / (params['m'] * params['omega']**2))
            else:
                x_turn = 1e-9

            x_lim = x_turn * 1.5
            x = np.linspace(-x_lim, x_lim, 500)
            
            U = 0.5 * params['m'] * params['omega']**2 * x**2
            ax.plot(x, U, 'k-', label='Потенціал U(x)')
            
            ax.hlines(E_n, -x_lim, x_lim, color='gray', linestyle='--', label=f'$E_{n_viz}$')
            
            alpha = np.sqrt(params['m'] * params['omega'] / HBAR)
            xi = alpha * x
            norm = 1 / np.sqrt(2**n_viz * math.factorial(n_viz)) * (alpha / np.pi**0.5)**0.5
            Hn = hermite(n_viz)
            psi = norm * np.exp(-xi**2 / 2) * Hn(xi)
            prob = psi**2
            
            scale = E_n * 0.5 if n_viz == 0 else (energies[1]-energies[0])
            
            ax.plot(x, E_n + (psi / np.max(np.abs(psi))) * scale, color='blue', label=r'$\Psi$')
            ax.plot(x, E_n + (prob / np.max(prob)) * scale, color='red', label=r'$|\Psi|^2$')
            ax.fill_between(x, E_n, E_n + (prob / np.max(prob)) * scale, alpha=0.1, color='red')
            
            draw_arrow(ax, -x_turn, x_turn, E_n, "2A")
            
            ax.set_ylim(0, energies[-1]*1.3)
            ax.legend(loc='upper right')
            st.pyplot(fig)

        # 4. СХОДИНКА
        elif sys_type == "Потенціальний Бар'єр" and sub_type == "Сходинка":
            res, R, T, k1, val2 = calc_step_coefficients(params['m'], params['E'], params['U0'])
            
            c1, c2 = st.columns([1, 2])
            with c1:
                if res == "Pass":
                    st.success("E > U₀: Проходження")
                    st.metric("T", f"{T:.4f}")
                    st.metric("R", f"{R:.4f}")
                elif res == "Reflect":
                    st.warning("E < U₀: Відбиття")
                    st.metric("R", "1.00")
                    st.write(f"Глибина: {val2:.2e} м")
                else:
                    st.error("Помилка в даних")
            
            with c2:
                fig, ax = plt.subplots(figsize=(8, 5))
                x = np.linspace(-2e-9, 2e-9, 500)
                U_viz = np.where(x>0, params['U0'], 0)
                
                ax.plot(x, U_viz, 'k-', linewidth=2, label='U(x)')
                ax.fill_between(x, U_viz, alpha=0.1, color='gray')
                ax.axhline(params['E'], color='orange', linestyle='--', label='E')
                
                if res == "Reflect":
                     x_tail = np.linspace(0, 2e-9, 100)
                     psi_tail = params['E'] + np.exp(-val2*x_tail) * (params['E']*0.2)
                     ax.plot(x_tail, psi_tail, color='green', label=r'Проникнення')

                ax.legend()
                st.pyplot(fig)

        # 5. ПРЯМОКУТНИЙ БАР'ЄР
        elif sys_type == "Потенціальний Бар'єр" and sub_type == "Прямокутний":
            T, R = calc_barrier_tunneling(params['m'], params['E'], params['U0'], params['L'])
            
            c1, c2 = st.columns([1, 2])
            with c1:
                if params['E'] < params['U0']:
                    st.info("Режим: Тунелювання")
                else:
                    st.success("Режим: Надбар'єрний")
                
                st.metric("T (Проходження)", f"{T:.4e}")
                st.metric("R (Відбиття)", f"{R:.4f}")
                
            with c2:
                fig, ax = plt.subplots(figsize=(8, 5))
                x = np.linspace(-params['L'], 2*params['L'], 500)
                U_viz = np.zeros_like(x)
                mask_bar = (x >= 0) & (x <= params['L'])
                U_viz[mask_bar] = params['U0']
                
                ax.plot(x, U_viz, 'k-', linewidth=2, label='Бар\'єр')
                ax.fill_between(x, U_viz, alpha=0.1, color='gray')
                ax.axhline(params['E'], color='red', linestyle='--', label='E')
                
                draw_arrow(ax, 0, params['L'], params['U0']*1.1, f"L={params['L']:.1e}")
                
                ax.legend()
                st.pyplot(fig)

if __name__ == "__main__":
    main()