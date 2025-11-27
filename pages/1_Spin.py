import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Налаштування сторінки
st.set_page_config(page_title="Спін і Сфера Блоха", layout="wide")

def run_spin_visualization():
    st.title("🌀 Квантовий Спін: Сфера Блоха")
    
    # Охайний опис без зайвого виділення
    st.markdown("""
    Спін електрона (або будь-якої дворівневої системи, наприклад, кубіта) можна представити як вектор на одиничній сфері:
    * **Північний полюс ($|0\\rangle$):** Спін направлений точно ВГОРУ (+Z).
    * **Південний полюс ($|1\\rangle$):** Спін направлений точно ВНИЗ (-Z).
    * **Екватор:** Стан суперпозиції (спін направлений вбік).
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎛 Параметри")
        st.info("Обертайте вектор, змінюючи кути:")
        
        # Слайдери для кутів
        theta = st.slider("Кут θ (Полярний)", 0.0, np.pi, 0.0, 0.01)
        phi = st.slider("Кут φ (Азимутальний)", 0.0, 2*np.pi, 0.0, 0.01)
        
        st.divider()
        st.subheader("📊 Стан системи")
        
        # Розрахунок амплітуд
        prob_up = np.cos(theta / 2) ** 2      
        prob_down = np.sin(theta / 2) ** 2    
        
        st.metric("Ймовірність Спін ВГОРУ (↑)", f"{prob_up * 100:.1f}%")
        st.metric("Ймовірність Спін ВНИЗ (↓)", f"{prob_down * 100:.1f}%")
        
        # Формула
        st.latex(r"|\Psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle")

    with col2:
        # --- 3D ВІЗУАЛІЗАЦІЯ ---
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        fig = go.Figure()
        
        # 1. Сфера (прозора)
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.1,
            showscale=False,
            colorscale='Blues',
            hoverinfo='skip'
        ))
        
        # 2. Осі координат (Товсті лінії)
        line_len = 1.1
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-line_len, line_len],
            mode='lines', line=dict(color='blue', width=5), name='Z'))
        fig.add_trace(go.Scatter3d(x=[-line_len, line_len], y=[0, 0], z=[0, 0],
            mode='lines', line=dict(color='green', width=5), name='X'))
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[-line_len, line_len], z=[0, 0],
            mode='lines', line=dict(color='orange', width=5), name='Y'))
        
        # 3. Вектор Спіна (Яскраво-червоний)
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines+markers',
            line=dict(color='red', width=10),     
            marker=dict(size=6, color='red'),    
            name='Вектор Спіна'
        ))
        
        # 4. Точка на поверхні
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond'),
            showlegend=False
        ))

        # 5. Текстові підписи полюсів
        fig.add_trace(go.Scatter3d(
            x=[0, 0, 1.3, 0, 0, 0],
            y=[0, 0, 0, 1.3, 0, 0],
            z=[1.1, -1.1, 0, 0, 0, 0],
            mode='text',
            text=['|0⟩ (↑)', '|1⟩ (↓)', '+X', '+Y', '', ''],
            textposition="top center",
            showlegend=False
        ))

        # Налаштування камери
        fig.update_layout(
            width=700, height=600,
            title="Інтерактивна Сфера Блоха",
            scene=dict(
                xaxis=dict(visible=False, showbackground=False),
                yaxis=dict(visible=False, showbackground=False),
                zaxis=dict(visible=False, showbackground=False),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_spin_visualization()