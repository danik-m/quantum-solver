import streamlit as st
import sys
import os

# --- МАГІЯ ДЛЯ ІМПОРТІВ (Виправляє Pylance/Module errors) ---
# Додаємо поточну папку до шляхів пошуку Python
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Тепер імпорти точно запрацюють
try:
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    from scipy import constants

    # Імпорт з ваших модулів
    from modules.utils import M_E, EV
    from modules.wells import (
        run_infinite_well_sim,
        run_finite_well_sim,
        get_wavefunction_inf,
        get_wavefunction_finite
    )
    from modules.oscillator import run_oscillator_sim
    from modules.barriers import run_barrier_sim
    from modules.wavepacket import run_wave_packet_simulation

except ImportError as e:
    st.error(f"🚨 Помилка імпорту! Перевірте структуру папок.\nДеталі: {e}")
    st.stop()

# --- НАЛАШТУВАННЯ СТОРІНКИ ---
st.set_page_config(layout="wide", page_title="Quantum Solver Modular", page_icon="⚛️")
plt.style.use('default') 

def main():
    st.sidebar.title("🎛 Панель Керування")
    
    # 1. Вибір системи
    sys_type = st.sidebar.selectbox("Оберіть задачу:", 
        ["Потенціальна Яма", "Потенціальний Бар'єр", "Гармонічний Осцилятор", "🌊 Хвильовий Пакет"])
    
    sub_type = None
    if sys_type == "Потенціальна Яма":
        sub_type = st.sidebar.radio("Тип:", ["Нескінченні стінки", "Кінцеві стінки"])
    elif sys_type == "Потенціальний Бар'єр":
        sub_type = st.sidebar.radio("Тип:", ["Сходинка", "Прямокутний бар'єр"])

    st.sidebar.markdown("---")
    
    # 2. Параметри
    st.sidebar.header("Параметри")
    params = {}
    
    # Вибір частинки
    p_name = st.sidebar.selectbox("Частинка:", ["Електрон", "Протон", "Мюон"])
    mass_map = {"Електрон": M_E, "Протон": constants.m_p, "Мюон": M_E * 207}
    params['m'] = mass_map[p_name]
    st.sidebar.caption(f"m = {params['m']:.2e} кг")

    # Динамічні поля
    if sys_type != "Гармонічний Осцилятор":
        params['L'] = st.sidebar.number_input(
    "Ширина L (м)",
    value=1e-9,
    step=1e-10,       # ← 0.1 нм
    format="%.2e"
)

    
    if sys_type in ["Потенціальний Бар'єр", "🌊 Хвильовий Пакет"] or \
       (sys_type == "Потенціальна Яма" and sub_type == "Кінцеві стінки"):
        params['U0'] = st.sidebar.number_input("Потенціал U₀ (еВ)", value=10.0, step=0.1) * EV
        
    if sys_type in ["Потенціальний Бар'єр", "🌊 Хвильовий Пакет"]:
        params['E'] = st.sidebar.number_input("Енергія E (еВ)", value=5.0, step=0.1) * EV
        
    if sys_type == "Гармонічний Осцилятор":
        params['omega'] = st.sidebar.number_input("Частота ω (рад/с)", value=5e15, format="%.2e", step=0.1e15)

    st.sidebar.markdown("---")
    
    # Кнопка запуску
    if st.sidebar.button("🚀 РОЗРАХУВАТИ", type="primary"):
        st.session_state['run_calc'] = True

    # 3. Запуск логіки з модулів
    if st.session_state.get('run_calc'):
        st.title(f"Результати: {sys_type}")
        
        if sys_type == "Потенціальна Яма":
            if sub_type == "Нескінченні стінки":
                run_infinite_well_sim(params)
            elif sub_type == "Кінцеві стінки":
                run_finite_well_sim(params)
                
        elif sys_type == "Гармонічний Осцилятор":
            run_oscillator_sim(params)
            
        elif sys_type == "Потенціальний Бар'єр":
            run_barrier_sim(params, sub_type)
            
        elif sys_type == "🌊 Хвильовий Пакет":
            run_wave_packet_simulation()

            import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.join(current_dir, "modules")

if modules_dir not in sys.path:
    sys.path.append(modules_dir)

if __name__ == "__main__":
    main()