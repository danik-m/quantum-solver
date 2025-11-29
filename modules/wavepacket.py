# modules/wavepacket.py
# Тепер хвиля ВЗАЄМОДІЄ з бар’єром — відбиття, тунелювання, резонанс!

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy import constants
from scipy.integrate import simps
import time

HBAR = constants.hbar
M_E = constants.m_e
EV = constants.electron_volt

def run_wave_packet_simulation():
    st.title("Хвильовий пакет + Багато бар’єрів")
    st.markdown("### Тепер хвиля ВЗАЄМОДІЄ з бар’єром — фізично правильно!")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Налаштування")
        energy_ev = st.slider("Енергія (еВ)", 20.0, 150.0, 60.0, 2.0)
        U0_ev = st.slider("Висота бар’єру (еВ)", 30.0, 200.0, 80.0, 5.0)
        width_nm = st.slider("Ширина бар’єру (нм)", 0.5, 6.0, 2.0, 0.2)
        gap_nm = st.slider("Відстань між бар’єрами (нм)", 2.0, 15.0, 6.0, 0.5)
        n_barriers = st.selectbox("Кількість бар’єрів", [1, 2, 3], index=1)
        fps = st.slider("Кадрів/с", 20, 60, 40)

        if st.button("ЗАПУСТИТИ", type="primary"):
            st.session_state.wp_running = True
            st.rerun()

    with col2:
        if st.session_state.get("wp_running"):
            with st.expander("Анімація — відкрито одразу", expanded=True):
                placeholder = st.empty()
                progress = st.progress(0)

                # Сітка
                Nx = 3000
                L = 2.2e-7
                x = np.linspace(-L/2, L/2, Nx)
                dx = x[1] - x[0]
                dt = 4e-18

                # Бар’єри
                V = np.zeros(Nx)
                width = width_nm * 1e-9
                gap = gap_nm * 1e-9
                total_span = n_barriers * width + max(0, n_barriers-1) * gap
                start_x = -total_span / 2 + width / 2

                barriers = []
                for i in range(n_barriers):
                    left = start_x + i * (width + gap)
                    right = left + width
                    mask = (x >= left) & (x <= right)
                    V[mask] = U0_ev * EV
                    barriers.append((left*1e9, right*1e9))

                # Початковий пакет
                k0 = np.sqrt(2 * M_E * energy_ev * EV) / HBAR
                x0 = -100e-9
                sigma = 6e-9
                psi = np.exp(-((x - x0)**2)/(4*sigma**2)) * np.exp(1j * k0 * x)
                psi /= np.sqrt(simps(np.abs(psi)**2, x))

                # ВИПРАВЛЕНИЙ Crank-Nicolson (це було головне!)
                r = 1j * HBAR * dt / (4 * M_E * dx**2)
                
                main_A = 1 + 2*r + 1j*dt*V/(2*HBAR)
                main_B = 1 - 2*r - 1j*dt*V/(2*HBAR)
                off = -r * np.ones(Nx-1)

                # КЛЮЧ: правильні знаки для B!
                A = diags([off, main_A, off], [-1, 0, 1], shape=(Nx, Nx), format='csc')
                B = diags([off, main_B, off], [-1, 0, 1], shape=(Nx, Nx), format='csc')  # off, а не -off!

                lu = splu(A)

                steps = 3500
                draw_every = max(1, steps // 200)

                for step in range(steps):
                    psi = lu.solve(B @ psi)

                    if step % draw_every == 0:
                        progress.progress(step / steps)

                        with placeholder.container():
                            fig = plt.figure(figsize=(16, 8))
                            fig.patch.set_facecolor('#0e1117')
                            ax = fig.add_subplot(111)
                            ax.set_facecolor('#0e1117')

                            # Потенціал
                            ax.plot(x*1e9, V/EV, color='lime', lw=6, label=f"{n_barriers} бар’єр(и) × {U0_ev:.0f} еВ")
                            ax.axhline(energy_ev, color='red', ls='--', lw=3, label=f"E = {energy_ev:.1f} еВ")

                            # Підсвітка
                            for l, r in barriers:
                                ax.axvspan(l, r, color='orange', alpha=0.4)

                            # Пакет
                            prob = np.abs(psi)**2
                            scale = 5e9
                            ax.plot(x*1e9, prob*scale, color='cyan', lw=4.5)
                            ax.fill_between(x*1e9, prob*scale, color='cyan', alpha=0.7)

                            ax.set_ylim(0, max(np.max(prob)*scale*1.5, U0_ev*1.2))
                            ax.set_xlim(-110, 110)
                            ax.set_xlabel("x (нм)", color='white', fontsize=14)
                            ax.set_title(f"t = {step*dt*1e15:.1f} фс | Хвиля взаємодіє з бар’єром!", 
                                       color='white', fontsize=18)
                            ax.tick_params(colors='white')
                            ax.legend(facecolor='#0e1117', labelcolor='white', fontsize=14)

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        time.sleep(1.0 / fps)

                st.success("Хвиля пройшла через бар’єр(и)!")
                st.balloons()
                st.session_state.wp_running = False

        else:
            st.info("Натисни «ЗАПУСТИТИ» — і побачиш справжню взаємодію!")

if __name__ == "__main__":
    run_wave_packet_simulation()