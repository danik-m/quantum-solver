import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Квантова Заплутаність", layout="wide")

def run_entanglement_simulation():
    st.title("🔗 Експеримент Белла: Квантова Заплутаність")
    
    # Вибір типу симуляції
    sim_type = st.radio("Оберіть демонстрацію:", 
        ["1. Візуалізація Штерна-Герлаха (3D Анімація)", 
         "2. Експеримент Белла (Графік кореляції)"])

    st.divider()

    # === 1. 3D ВІЗУАЛІЗАЦІЯ ШТЕРНА-ГЕРЛАХА ===
    if sim_type == "1. Візуалізація Штерна-Герлаха (3D Анімація)":
        st.subheader("Експеримент з парою заплутаних електронів")
        st.markdown("""
        **Опис процесу:**
        1.  Джерело випускає пару електронів із сумарним спіном 0.
        2.  Вони розлітаються в протилежні сторони до магнітів Штерна-Герлаха.
        3.  Магнітне поле розщеплює пучок: електрони відхиляються або до **N**, або до **S**.
        4.  Через заплутаність, якщо один електрон летить до **N**, інший *обов'язково* летить до **S**.
        """)
        
        if st.button("🔴 Запустити пару електронів"):
            outcome = np.random.choice([0, 1]) 
            steps = 50
            x_left = np.linspace(0, -4, steps)
            x_right = np.linspace(0, 4, steps)
            y = np.zeros(steps)
            z_left = np.zeros(steps)
            z_right = np.zeros(steps)
            split_idx = int(steps * 0.5) 
            
            if outcome == 0: 
                z_left[split_idx:] = np.linspace(0, 1, steps - split_idx)**2
                z_right[split_idx:] = np.linspace(0, -1, steps - split_idx)**2
                res_text = "Результат: Лівий ⬆ (до S), Правий ⬇ (до N)"
                color_left, color_right = 'red', 'blue'
            else:
                z_left[split_idx:] = np.linspace(0, -1, steps - split_idx)**2
                z_right[split_idx:] = np.linspace(0, 1, steps - split_idx)**2
                res_text = "Результат: Лівий ⬇ (до N), Правий ⬆ (до S)"
                color_left, color_right = 'blue', 'red'

            fig = go.Figure()

            # Джерело
            fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=15, color='purple'), name='Джерело'))

            # Магніти (спрощено)
            fig.add_trace(go.Mesh3d(x=[-2,-2,-2,-2], y=[-1,1,1,-1], z=[1,1,-1,-1], color='gray', name='Магніт L', opacity=0.3))
            fig.add_trace(go.Mesh3d(x=[2,2,2,2], y=[-1,1,1,-1], z=[1,1,-1,-1], color='gray', name='Магніт R', opacity=0.3))

            # Анімація
            frames = []
            for i in range(steps):
                frames.append(go.Frame(data=[
                    go.Scatter3d(x=[x_left[i]], y=[0], z=[z_left[i]], mode='markers', marker=dict(color=color_left, size=8)),
                    go.Scatter3d(x=[x_right[i]], y=[0], z=[z_right[i]], mode='markers', marker=dict(color=color_right, size=8))
                ]))

            fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(color=color_left, size=8), name='E- L'))
            fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(color=color_right, size=8), name='E- R'))

            fig.update_layout(
                width=800, height=500, title="3D Симуляція",
                scene=dict(xaxis=dict(range=[-5, 5]), zaxis=dict(range=[-2, 2])),
                updatemenus=[dict(type="buttons", buttons=[dict(label="▶️ Старт", method="animate", args=[None, dict(frame=dict(duration=30), fromcurrent=True)])])]
            )
            fig.frames = frames
            st.plotly_chart(fig)
            st.success(f"**{res_text}**")

    # === 2. ЕКСПЕРИМЕНТ БЕЛЛА ===
    elif sim_type == "2. Експеримент Белла (Графік кореляції)":
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("🛠 Налаштування")
            angle_a = st.slider("Кут Аліси (α)", 0, 360, 0, step=15)
            angle_b = st.slider("Кут Боба (β)", 0, 360, 45, step=15)
            theta_deg = abs(angle_a - angle_b)
            st.info(f"Різниця кутів: **{theta_deg}°**")
            
        with c2:
            st.subheader("🎲 Симуляція")
            n_shots = st.select_slider("Кількість вимірювань", [100, 1000, 5000], value=1000)
            if st.button("Запустити"):
                theta_rad = np.radians(theta_deg)
                prob_diff = np.cos(theta_rad/2)**2
                random_vals = np.random.random(n_shots)
                diff_count = np.sum(random_vals < prob_diff)
                same_count = n_shots - diff_count
                corr_exp = (same_count - diff_count) / n_shots
                
                fig, ax = plt.subplots(figsize=(6, 3))
                angles = np.linspace(0, 360, 100)
                ax.plot(angles, -np.cos(np.radians(angles)), 'k--', label='Теорія')
                ax.plot([theta_deg], [corr_exp], 'ro', label='Результат')
                ax.set_xlabel("Різниця кутів"); ax.set_ylabel("Кореляція")
                ax.legend()
                st.pyplot(fig)

if __name__ == "__main__":
    run_entanglement_simulation()