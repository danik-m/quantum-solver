import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go # Нова бібліотека для 3D
from scipy import constants
from scipy.special import hermite

# --- 1. КОНСТАНТИ ---
HBAR = constants.hbar
M_E = constants.m_e
EV = constants.electron_volt

# --- 2. МАТЕМАТИЧНИЙ ДВИГУН (ЯМИ ТА БАР'ЄРИ) ---

def calc_infinite_well_energy(m, L, n):
    return (n**2 * np.pi**2 * HBAR**2) / (2 * m * L**2)

def calc_harmonic_energy(omega, n):
    return HBAR * omega * (n + 0.5)

def calc_step_coefficients(m, E, U0):
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
    if U0 <= 0: return 0, 0
    z0 = (L / 2) * np.sqrt(2 * m * U0) / HBAR
    N = 1 + int((2 * z0) / np.pi)
    return N, z0

# --- 3. ФУНКЦІЇ ВІЗУАЛІЗАЦІЇ (HELPER) ---

def draw_arrow(ax, x1, x2, y, text, color='black'):
    ax.annotate('', xy=(x1, y), xytext=(x2, y), arrowprops=dict(arrowstyle='<->', color=color))
    ax.text((x1+x2)/2, y, text, ha='center', va='bottom', color=color, 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# --- 4. НОВИЙ БЛОК: СПІН (SFЁRA БЛОХА) ---

def run_spin_visualization():
    st.markdown("# 🌀 Візуалізація Спіна (Сфера Блоха)")
    st.markdown("""
    Тут ми розглядаємо спін електрона (або будь-яку дворівневу квантову систему, кубіт).
    Будь-який чистий стан спіна $|\psi\rangle$ можна зобразити як точку на поверхні сфери одиничного радіуса.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎛 Параметри Спіна")
        # Кути сферичних координат
        theta = st.slider("Кут θ (Полярний)", 0.0, np.pi, 0.0, step=0.01, 
                          help="Визначає ймовірність виміряти спін ВГОРУ (0) або ВНИЗ (pi).")
        
        phi = st.slider("Кут φ (Азимутальний)", 0.0, 2*np.pi, 0.0, step=0.01,
                        help="Визначає фазу квантового стану (обертання навколо осі Z).")
        
        st.markdown("---")
        st.markdown("### 📊 Стан системи")
        
        # Розрахунок амплітуд ймовірності
        # |psi> = cos(theta/2)|0> + e^(i*phi)*sin(theta/2)|1>
        a_real = np.cos(theta/2)
        b_magn = np.sin(theta/2)
        
        # Ймовірності
        prob_up = a_real**2
        prob_down = b_magn**2
        
        st.metric("Ймовірність Спін ВГОРУ (↑)", f"{prob_up*100:.1f}%")
        st.metric("Ймовірність Спін ВНИЗ (↓)", f"{prob_down*100:.1f}%")
        
        # Векторне представлення
        st.latex(r"|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle")
        
    with col2:
        # --- 3D ВІЗУАЛІЗАЦІЯ (PLOTLY) ---
        
        # Перетворення сферичних координат в Декартові
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        # Створення сфери (сітка)
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig = go.Figure()
        
        # 1. Напівпрозора сфера
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.1,
            showscale=False,
            colorscale='Blues',
            hoverinfo='skip'
        ))
        
        # 2. Вектор спіна (Стрілка)
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines+markers',
            line=dict(color='red', width=10),
            marker=dict(size=5, color='red'),
            name='Вектор спіна'
        ))
        
        # 3. Точка на поверхні
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Стан'
        ))
        
        # 4. Осі координат
        axis_length = 1.2
        # Вісь Z (синя)
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-axis_length, axis_length], mode='lines', line=dict(color='blue', width=2), name='Z'))
        # Вісь X (зелена)
        fig.add_trace(go.Scatter3d(x=[-axis_length, axis_length], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='green', width=2), name='X'))
        # Вісь Y (помаранчева)
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[-axis_length, axis_length], z=[0, 0], mode='lines', line=dict(color='orange', width=2), name='Y'))

        # 5. Підписи полюсів
        fig.add_trace(go.Scatter3d(
            x=[0, 0, 1.3, 0, 0], 
            y=[0, 0, 0, 1.3, 0], 
            z=[1.1, -1.1, 0, 0, 0],
            mode='text',
            text=['|0⟩ (↑)', '|1⟩ (↓)', '+X', '+Y', 'Центр'],
            textposition="top center",
            showlegend=False
        ))

        # Налаштування макету
        fig.update_layout(
            width=700, height=600,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data' # Щоб сфера була круглою, а не сплюснутою
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            title="Інтерактивна Сфера Блоха (Можна крутити мишкою!)"
        )
        
        st.plotly_chart(fig)

# --- 5. ГОЛОВНИЙ ІНТЕРФЕЙС ---

def main():
    st.set_page_config(layout="wide", page_title="Quantum Physics Solver")
    
    if 'calc_active' not in st.session_state:
        st.session_state['calc_active'] = False

    st.title("⚛️ Квантовий Розв'язувач")

    # --- САЙДБАР (ГОЛОВНЕ МЕНЮ) ---
    st.sidebar.header("1. Головне Меню")
    
    # Додано "Спін" в меню
    main_mode = st.sidebar.radio("Оберіть розділ:", 
                                 ["Задачі (Ями та Бар'єри)", "🌀 Спін (Нове!)"])
    
    # ================== РОЗДІЛ СПІНА (НОВИЙ) ==================
    if main_mode == "🌀 Спін (Нове!)":
        run_spin_visualization()
        return # Виходимо з функції, щоб не малювати задачі
        
    # ================== РОЗДІЛ ЗАДАЧ (СТАРИЙ, ПЕРЕВІРЕНИЙ) ==================
    
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
    
    # Словник параметрів
    params = {}
    # ВИПРАВЛЕННЯ: Вибір частинки (для маси)
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
                
                # Лінія рівня (пунктир)
                ax.hlines(E_cur, 0, params['L'], color='gray', linestyle='--', label=f'$E_{n_viz}$')
                
                x = np.linspace(0, params['L'], 300)
                psi = np.sqrt(2/params['L']) * np.sin(n_viz * np.pi * x / params['L'])
                prob = psi**2
                scale = E_cur * 0.5 
                
                # Графіки як на скріншоті
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