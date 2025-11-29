import streamlit as st
import time

# --- CONSTANTS & PHYSICS ---
C = 300000  # Speed of light in km/s (scale)
V = 240000  # Train speed in km/s
GAMMA = 5 / 3  # 1.6666...
L0_KM = 8.64 * 10**8  # Proper length in km

# Initial Time on clocks at first meeting (12:00:00)
T0_SECONDS = 12 * 3600

# Contracted Length
L_CONTRACTED = L0_KM / GAMMA

# Events (in seconds from T0 in Platform Frame K)
T_EVENT_1 = 0
T_EVENT_2 = L_CONTRACTED / V  # 2160 sec
T_EVENT_3 = L0_KM / V  # 3600 sec

# --- HELPER FUNCTIONS ---

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

def get_train_time(t_k, x_k):
    """
    Lorentz Transformation for Time.
    t, x are in Frame K (Platform). We want t' in Frame K' (Train)
    t' = gamma * (t - v*x / c^2)
    """
    delta = (V * x_k) / (C**2)
    return GAMMA * (t_k - delta)

def get_modal_content(step):
    if step == 1:
        return {
            "title": "Подія 1: Старт",
            "desc": "Ніс поїзда (B') зрівнявся з початком платформи (A).",
            "calc": "Це момент синхронізації. Ми приймаємо t=12:00:00. Спостерігач у поїзді бачить, що годинник B (попереду) відстає і показує 10:56:00.",
            "next_step": 2,
            "next_text": "Далі: чекаємо хвоста поїзда"
        }
    elif step == 2:
        return {
            "title": "Подія 2: Хвіст на старті",
            "desc": "Хвіст поїзда (A') порівнявся з початком платформи (A).",
            "calc": "На платформі пройшло 36 хв (t=12:36). Але на годиннику хвоста поїзда 13:00. Хвіст 'думає', що пройшло більше часу.",
            "next_step": 3,
            "next_text": "Перейти на кінець платформи"
        }
    elif step == 3:
        return {
            "title": "Подія 3: Голова на фініші",
            "desc": "Ніс поїзда (B') дістався кінця платформи (B).",
            "calc": "На платформі пройшла 1 година (t=13:00). На годиннику поїзда лише 12:36. Час у рухомому поїзді сповільнився.",
            "next_step": 4,
            "next_text": "Змінити систему: Перейти в Поїзд"
        }
    elif step == 4:
        return {
            "title": "Система Поїзда: Старт",
            "desc": "Ми в голові поїзда (B'). Платформа летить на нас.",
            "calc": "Годинник A показує 12:00, а годинник B (біля нас) показує 10:56. Платформа скорочена.",
            "next_step": 5,
            "next_text": "Їдемо до кінця платформи"
        }
    elif step == 5:
        return {
            "title": "Система Поїзда: Фініш голови",
            "desc": "Ми проїхали кінець платформи (B).",
            "calc": "На нашому годиннику 12:36. На годиннику B теж 12:36 (бо він відставав з початку).",
            "next_step": 6,
            "next_text": "Перейти у хвіст поїзда"
        }
    elif step == 6:
        return {
            "title": "Система Поїзда: Хвіст",
            "desc": "Ми у хвості. Платформа пролітає повз.",
            "calc": "Коли початок платформи (A) порівнявся з нами: наш час 13:00, час платформи A 12:36. Час платформи йде повільніше для нас.",
            "next_step": 0,
            "next_text": "Завершити"
        }
    return None

# --- STREAMLIT APP ---

st.set_page_config(page_title="Relativity Simulator", page_icon="🚄", layout="wide")

# Initialize Session State
if 'scenario_step' not in st.session_state:
    st.session_state.scenario_step = 0
if 'time_k' not in st.session_state:
    st.session_state.time_k = -1000.0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# Constants for Animation
SIM_SPEED_MULTIPLIER = 500  # Sim seconds per Real second
FPS = 30
DT = 1.0 / FPS

# --- LOGIC CONTROL ---

def reset_sim(step):
    st.session_state.scenario_step = step
    st.session_state.time_k = -1000.0
    st.session_state.is_running = False

# Step Logic
step = st.session_state.scenario_step
is_train_frame = step >= 4

# Physics Update Loop (runs when 'is_running' is True)
if st.session_state.is_running:
    # Update time
    st.session_state.time_k += SIM_SPEED_MULTIPLIER * DT
    
    # Check Events
    stop_time = None
    if (step == 1 or step == 4) and st.session_state.time_k >= T_EVENT_1:
        stop_time = T_EVENT_1
    elif (step == 2 or step == 6) and st.session_state.time_k >= T_EVENT_2:
        stop_time = T_EVENT_2
    elif (step == 3 or step == 5) and st.session_state.time_k >= T_EVENT_3:
        stop_time = T_EVENT_3
    
    if stop_time is not None:
        st.session_state.time_k = stop_time
        st.session_state.is_running = False
        st.rerun()
    
    time.sleep(DT)
    st.rerun()

# --- UI RENDERING ---

st.title("🚄 Симулятор СТВ: Поїзд Ейнштейна")
st.caption(f"v = 240,000 км/с (0.8c) | γ = {GAMMA:.3f}")

# Header Controls
col1, col2 = st.columns([3, 1])
with col1:
    if step == 0:
        st.info("👋 Ласкаво просимо! Натисніть кнопку нижче, щоб почати експеримент.")
        if st.button("Почати Експеримент", type="primary"):
            reset_sim(1)
            st.rerun()
    else:
        # Check if we are stopped at an event
        modal_info = None
        if not st.session_state.is_running:
            # Check precise timing match for event
            if (step == 1 or step == 4) and abs(st.session_state.time_k - T_EVENT_1) < 1: modal_info = get_modal_content(step)
            elif (step == 2 or step == 6) and abs(st.session_state.time_k - T_EVENT_2) < 1: modal_info = get_modal_content(step)
            elif (step == 3 or step == 5) and abs(st.session_state.time_k - T_EVENT_3) < 1: modal_info = get_modal_content(step)

        if modal_info:
            st.success(f"**{modal_info['title']}**")
            st.markdown(f"{modal_info['desc']}")
            st.warning(f"📐 {modal_info['calc']}")
            if st.button(f"{modal_info['next_text']} ➡️"):
                reset_sim(modal_info['next_step'])
                if modal_info['next_step'] in [1, 4]: # Auto start only on new systems
                     st.session_state.is_running = False 
                else:
                     st.session_state.is_running = False # Wait for user to play? Or auto? Let's wait.
                st.rerun()
        else:
            st.write(f"👁️ **Спостерігач:** {'В Поїзді (K\')' if is_train_frame else 'На Платформі (K)'}")
            
with col2:
    if step > 0:
        t_display = format_time(T0_SECONDS + st.session_state.time_k) if st.session_state.time_k >= 0 else "Наближення..."
        st.metric("Системний Час (K)", t_display)
        
        c_play, c_reset = st.columns(2)
        if c_play.button("⏯️ Старт/Пауза"):
            st.session_state.is_running = not st.session_state.is_running
            st.rerun()
        if c_reset.button("🔄 Скидання"):
            st.session_state.time_k = -1000.0
            st.session_state.is_running = False
            st.rerun()

# --- VISUALIZATION (SVG via HTML) ---
if step > 0:
    # Calculations for View
    current_time_k = st.session_state.time_k
    head_pos_km = V * current_time_k
    tail_pos_km = head_pos_km - L_CONTRACTED
    
    # Viewport Mapping (0 -> 10%, L0 -> 90%)
    def km_to_pct(km):
        return 10 + (km / L0_KM) * 80
    
    train_width_pct = 80 * (1/GAMMA) # ~48%
    
    # Train Position Logic
    if is_train_frame:
        # Simplified Visuals for Train Frame: Platform moves left.
        platform_left_style = f"{10 - ((V * current_time_k) / L0_KM * 80)}%"
        train_left_style = "26%" # Fixed
        
    else:
        # Platform Static
        platform_left_style = "10%"
        # Train Moves Right
        train_left_pct = km_to_pct(tail_pos_km)
        train_left_style = f"{train_left_pct}%"

    # Clocks
    clock_a = T0_SECONDS + current_time_k
    clock_b = T0_SECONDS + current_time_k
    
    t_prime_head = T0_SECONDS + get_train_time(current_time_k, head_pos_km)
    t_prime_tail = T0_SECONDS + get_train_time(current_time_k, tail_pos_km)
    
    # SVG Content - NO INDENTATION ALLOWED FOR HTML TAGS
    svg_html = f"""
<div style="background-color: #1e293b; border-radius: 10px; padding: 20px; position: relative; height: 300px; overflow: hidden; border: 4px solid #334155;">
<div style="position: absolute; inset: 0; opacity: 0.2; background-image: linear-gradient(#4f46e5 1px, transparent 1px), linear-gradient(90deg, #4f46e5 1px, transparent 1px); background-size: 40px 40px;"></div>
<div style="position: absolute; top: 50%; left: 0; width: 100%; height: 2px; background: rgba(30, 64, 175, 0.5); transform: translateY(10px);"></div>
<div style="position: absolute; top: 50%; left: {platform_left_style}; width: 80%; height: 40px; background: #2563eb; border-bottom: 4px solid #1e40af; transform: translateY(20px); transition: left 0.05s linear; display: flex; justify-content: space-between; align-items: flex-end; padding: 0 10px;">
<div style="color: #60a5fa; font-weight: bold; font-family: sans-serif; position: absolute; top: 100%; width: 100%; text-align: center; margin-top: 5px;">ПЛАТФОРМА</div>
<div style="position: relative; top: -50px; left: -10px; background: #0f172a; border: 2px solid #3b82f6; padding: 4px; border-radius: 4px; text-align: center; width: 80px;">
<div style="color: #60a5fa; font-size: 10px; font-weight: bold;">A (Плат)</div>
<div style="color: white; font-family: monospace;">{format_time(clock_a)}</div>
<div style="width: 2px; height: 20px; background: #3b82f6; margin: 0 auto;"></div>
</div>
<div style="position: relative; top: -50px; right: -10px; background: #0f172a; border: 2px solid #3b82f6; padding: 4px; border-radius: 4px; text-align: center; width: 80px;">
<div style="color: #60a5fa; font-size: 10px; font-weight: bold;">B (Плат)</div>
<div style="color: white; font-family: monospace;">{format_time(clock_b)}</div>
<div style="width: 2px; height: 20px; background: #3b82f6; margin: 0 auto;"></div>
</div>
</div>
<div style="position: absolute; top: 50%; left: {train_left_style}; width: {train_width_pct}%; height: 50px; background: #dc2626; border-bottom: 4px solid #991b1b; transform: translateY(-30px); border-radius: 8px; transition: left 0.05s linear; display: flex; justify-content: space-between; align-items: flex-start; padding: 0 10px; z-index: 10;">
<div style="color: #f87171; font-weight: bold; font-family: sans-serif; position: absolute; bottom: 100%; width: 100%; text-align: center; margin-bottom: 5px;">ПОЇЗД</div>
<div style="position: relative; top: -60px; left: -10px; background: #0f172a; border: 2px solid #ef4444; padding: 4px; border-radius: 4px; text-align: center; width: 80px;">
<div style="color: #ef4444; font-size: 10px; font-weight: bold;">A' (Поїзд)</div>
<div style="color: white; font-family: monospace;">{format_time(t_prime_tail)}</div>
<div style="width: 2px; height: 20px; background: #ef4444; margin: 0 auto;"></div>
</div>
<div style="position: relative; top: -60px; right: -10px; background: #0f172a; border: 2px solid #ef4444; padding: 4px; border-radius: 4px; text-align: center; width: 80px;">
<div style="color: #ef4444; font-size: 10px; font-weight: bold;">B' (Поїзд)</div>
<div style="color: white; font-family: monospace;">{format_time(t_prime_head)}</div>
<div style="width: 2px; height: 20px; background: #ef4444; margin: 0 auto;"></div>
</div>
</div>
</div>
"""
    
    st.markdown(svg_html, unsafe_allow_html=True)
    
    # Legend
    st.markdown("""
    <div style="text-align: right; color: gray; font-size: 12px; margin-top: 5px;">
    🟦 Годинники Платформи (K) | 🟥 Годинники Поїзда (K')
    </div>
    """, unsafe_allow_html=True)

# --- SOLUTION TEXT ---
with st.expander("📖 Детальне пояснення та розрахунки", expanded=True):
    st.markdown(r"""
    ### 1. Вихідні Дані
    * **Швидкість ($v$):** 240,000 км/с ($0.8c$)
    * **Фактор Лоренца ($\gamma$):** $5/3 \approx 1.67$
    * **Власна довжина ($L_0$):** $8.64 \cdot 10^8$ км
    * **Скорочена довжина ($L$):** $5.184 \cdot 10^8$ км

    ### 2. Ключові ефекти
    1.  **Відносність одночасності:** Події, одночасні в системі платформи ($t_A=t_B$), не є одночасними в системі поїзда.
        * Різниця часу: $\Delta t' = v L_0 / c^2$.
        * У момент зустрічі ($t=12:00$), спостерігач на поїзді бачить на годиннику $B$ час **10:56**.
    
    2.  **Уповільнення часу:**
        * Годинник, що рухається, йде повільніше у $\gamma$ разів.
        * Коли на платформі проходить 1 година, на годиннику поїзда проходить лише 36 хвилин.
    """)