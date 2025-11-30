# modules/wavepacket_fixed.py
# Виправлений та робочий варіант симулятора хвильового пакета, взаємодія з бар'єрами.
# Основні виправлення:
# - Убрані подвійні/конфліктні імпорти
# - Виправлені знаки в матрицях Crank-Nicolson (A та B)
# - Додано прості Dirichlet-умови на границях (psi=0)
# - Додано опціональну поглинаючу зону на краях (щоб уникати штучних відбиттів)
# - Виправлена нормалізація пакету та масштаб відображення ймовірності

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy import constants
from scipy.integrate import simps
import time

# Фізичні константи
HBAR = constants.hbar
M_E = constants.m_e
EV = constants.electron_volt

# Головна функція
def run_wave_packet_simulation():
    st.set_page_config(layout="wide")
    st.title("Хвильовий пакет + Багато бар’єрів — виправлено")
    st.markdown("### Тепер хвиля ВЗАЄМОДІЄ з бар’єром — фізично правильно!")

    # Гарантуємо, що ключ існує
    if "wp_running" not in st.session_state:
        st.session_state.wp_running = False

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Налаштування")
        energy_ev = st.slider("Енергія (еВ)", 1.0, 300.0, 60.0, 1.0)
        U0_ev = st.slider("Висота бар’єру (еВ)", 0.0, 300.0, 80.0, 1.0)
        width_nm = st.slider("Ширина бар’єру (нм)", 0.5, 10.0, 2.0, 0.1)
        gap_nm = st.slider("Відстань між бар’єрами (нм)", 1.0, 30.0, 6.0, 0.5)
        n_barriers = st.selectbox("Кількість бар’єрів", [1, 2, 3, 4], index=1)
        fps = st.slider("Кадрів/с", 5, 60, 30)
        use_absorber = st.checkbox("Використовувати поглинаючу зону на краях (рекомендується)", value=True)

        if st.button("ЗАПУСТИТИ", type="primary"):
            st.session_state.wp_running = True
            st.rerun()


    with col2:
        if st.session_state.get("wp_running"):
            with st.expander("Анімація — відкрито одразу", expanded=True):
                placeholder = st.empty()
                progress = st.progress(0.0)

                # Сітка
                Nx = 3000
                L = 2.2e-7  # повна довжина (м) -> -110 нм .. +110 нм
                x = np.linspace(-L/2, L/2, Nx)
                dx = x[1] - x[0]

                # Часовий крок
                dt = 4e-18

                # Побудова потенціалу
                V = np.zeros(Nx, dtype=np.complex128)
                width = width_nm * 1e-9
                gap = gap_nm * 1e-9
                total_span = n_barriers * width + max(0, n_barriers - 1) * gap
                start_x = -total_span / 2 + width / 2

                barriers = []
                for i in range(n_barriers):
                    left = start_x + i * (width + gap)
                    right = left + width
                    mask = (x >= left) & (x <= right)
                    V[mask] = U0_ev * EV
                    barriers.append((left * 1e9, right * 1e9))

                # Поглинаюча зона (комплексна потенціальна частина) — плавно росте до країв
                if use_absorber:
                    absorb_width = 20e-9  # ширина поглинача з кожного краю
                    gamma_max = 1e-17  # сила поглинання (підбирайте)
                    left_mask = x < (x[0] + absorb_width)
                    right_mask = x > (x[-1] - absorb_width)

                    # масштабована квадратична функція для м'якого поглинання
                    if np.any(left_mask):
                        dist_left = (x[left_mask] - (x[0])) / absorb_width
                        V[left_mask] += -1j * gamma_max * (1 - np.cos(np.pi * dist_left))**2
                    if np.any(right_mask):
                        dist_right = ((x[right_mask]) - (x[-1] - absorb_width)) / absorb_width
                        V[right_mask] += -1j * gamma_max * (1 - np.cos(np.pi * dist_right))**2

                # Початковий пакет
                k0 = np.sqrt(2 * M_E * energy_ev * EV) / HBAR
                x0 = -100e-9
                sigma = 6e-9
                psi = np.exp(-((x - x0) ** 2) / (4 * sigma ** 2)) * np.exp(1j * k0 * x)
                # нормалізація
                psi /= np.sqrt(simps(np.abs(psi) ** 2, x))

                # Crank-Nicolson: правильні коефіцієнти
                # коэффициент схемы
                r = 1j * HBAR * dt / (2 * M_E * dx**2)

                # главные диагонали
                main_A = 1 + 2*r + 1j*dt*V/(2*HBAR)
                main_B = 1 - 2*r - 1j*dt*V/(2*HBAR)

                # боковые диагонали (общие!)
                off = -r

                # матрицы схемы
                A = diags([off*np.ones(Nx-1), main_A, off*np.ones(Nx-1)], [-1,0,1], format="csc")
                B = diags([off*np.ones(Nx-1), main_B, off*np.ones(Nx-1)], [-1,0,1], format="csc")

                lu = splu(A)


                # Dirichlet BC: psi[0] = psi[-1] = 0 (щоб уникнути проблем зі стабільністю на межах)
                # Переключаємо в LIL щоб просто змінити рядки
                A = A.tolil()
                B = B.tolil()

                
                A[-1, :] = 0
                A[-1, -1] = 1

                B[0, :] = 0
                B[0, 0] = 1
                B[-1, :] = 0
                B[-1, -1] = 1

                A = A.tocsc()
                B = B.tocsc()

                # Розклад для багаторазового вирішення системи
                lu = splu(A)

                steps = 3500
                draw_every = max(1, steps // 200)

                for step in range(steps):
                    # обчислюємо наступний крок
                    rhs = B.dot(psi)
                    psi = lu.solve(rhs)

                    # нормалізація (через невеликі чисельні похибки)
                    if step % 50 == 0:
                        norm = simps(np.abs(psi) ** 2, x)
                        psi /= np.sqrt(norm)

                    if step % draw_every == 0:
                        progress.progress(min(1.0, step / steps))

                        with placeholder.container():
                            fig = plt.figure(figsize=(12, 6))
                            fig.patch.set_facecolor('#0e1117')
                            ax = fig.add_subplot(111)
                            ax.set_facecolor('#0e1117')

                            # Потенціал в еВ (реальна частина)
                            ax.plot(x * 1e9, np.real(V) / EV, lw=3, label=f"{n_barriers} бар’єр(и) × {U0_ev:.0f} еВ")
                            ax.axhline(energy_ev, color='red', ls='--', lw=2, label=f"E = {energy_ev:.1f} еВ")

                            # Підсвітка бар'єрів
                            for l, r_ in barriers:
                                ax.axvspan(l, r_, color='orange', alpha=0.35)

                            # Пакет — ймовірність
                            prob = np.abs(psi) ** 2
                            # Масштабуємо ймовірність для візуалізації: нормалізуємо до половини висоти потенціалу
                            if prob.max() > 0:
                                prob_plot = prob / prob.max() * max(0.5, U0_ev / 2.0)
                            else:
                                prob_plot = prob

                            ax.plot(x * 1e9, prob_plot, color='cyan', lw=2.5)
                            ax.fill_between(x * 1e9, prob_plot, color='cyan', alpha=0.5)

                            ax.set_ylim(0, max(np.max(prob_plot) * 1.5, U0_ev * 1.2))
                            ax.set_xlim(x[0] * 1e9, x[-1] * 1e9)
                            ax.set_xlabel("x (нм)", color='white', fontsize=12)
                            ax.set_title(f"t = {step * dt * 1e15:.2f} фс | Хвиля взаємодіє з бар’єром!", color='white')
                            ax.tick_params(colors='white')
                            ax.legend(facecolor='#0e1117', labelcolor='white')

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        time.sleep(1.0 / fps)

                st.success("Хвиля пройшла через бар’єр(и) або взаємодія завершена!")
                st.balloons()
                st.session_state.wp_running = False

        else:
            st.info("Натисни «ЗАПУСТИТИ» — і побачиш справжню взаємодію!")


if __name__ == "__main__":
    run_wave_packet_simulation()
