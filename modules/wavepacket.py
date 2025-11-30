import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import splu
# Імпортуємо ТІЛЬКИ те, що реально є в utils.py
from .utils import HBAR, EV, plot_setup, M_E

class TimeDependentSolver:
    def __init__(self, m, Nx=400, L_space=4e-8):
        self.m = m
        self.Nx = Nx
        self.L_space = L_space
        self.x = np.linspace(0, L_space, Nx)
        self.dx = self.x[1] - self.x[0]
        self.dt = 1e-17 

    def simulate_packet(self, E_kin, U0, barrier_width):
        V = np.zeros(self.Nx)
        center = int(self.Nx * 0.5)
        w_idx = int(barrier_width / self.dx)
        V[center : center + w_idx] = U0
        
        # Хвильовий вектор k0
        k0 = np.sqrt(2 * self.m * E_kin) / HBAR
        x0 = self.L_space * 0.2
        sigma = self.L_space * 0.05
        
        # Початковий хвильовий пакет (Гаусс)
        psi = np.exp(-0.5 * ((self.x - x0)/sigma)**2) * np.exp(1j * k0 * self.x)
        
        # НОРМАЛІЗАЦІЯ (Виправлено: використовуємо np.trapz замість simps)
        norm = np.sqrt(np.trapz(np.abs(psi)**2, self.x))
        if norm > 0:
            psi /= norm
        
        # Побудова Гамільтоніана (Crank-Nicolson)
        h_val = HBAR**2 / (2 * self.m * self.dx**2)
        diag = np.full(self.Nx, 2*h_val) + V
        off = np.full(self.Nx-1, -h_val)
        H = sparse.diags([off, diag, off], [-1, 0, 1], shape=(self.Nx, self.Nx))
        
        factor = 1j * self.dt / (2 * HBAR)
        self.A = sparse.eye(self.Nx) + factor * H
        self.B = sparse.eye(self.Nx) - factor * H
        
        return self.x, psi, self.A, self.B, V

def run_wave_packet_simulation(params):
    # Додаємо ключ, щоб кнопка не конфліктувала
    if st.button("▶️ Старт", key="wp_start_btn"):
        solver = TimeDependentSolver(params['m'], Nx=400, L_space=4e-8)
        # Запускаємо симуляцію
        x, psi, A, B, V = solver.simulate_packet(params['E'], params['U0'], barrier_width=2e-9)
        
        plot_spot = st.empty()
        
        # Підготовка розв'язувача для прискорення
        try: 
            lu = splu(A.tocsc())
        except: 
            lu = None
            
        # Цикл анімації
        for i in range(80):
            rhs = B.dot(psi)
            if lu:
                psi = lu.solve(rhs)
            else:
                psi = sparse.linalg.spsolve(A, rhs)
                
            # Малюємо кожен 2-й кадр
            if i % 2 == 0:
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # Використовуємо plot_setup з utils
                try:
                    plot_setup(ax, f"Часова еволюція t={i}", xlabel="x (м)", ylabel="|Psi|^2")
                except:
                    ax.set_title(f"t={i}")
                    ax.grid(True)
                
                # Масштабування потенціалу для красивого відображення
                max_psi = np.max(np.abs(psi)**2)
                scale_V = max_psi / (np.max(V/EV) + 1e-30) if np.max(V)>0 else 0
                
                # Малюємо бар'єр і хвилю
                ax.plot(x*1e9, (V/EV) * scale_V, color='gray', alpha=0.5, label="Бар'єр")
                ax.plot(x*1e9, np.abs(psi)**2, color='cyan', lw=2, label="Хвиля")
                
                ax.set_ylim(0, max_psi * 1.5)
                plot_spot.pyplot(fig)
                plt.close(fig)

# Функція для автономного запуску (якщо запускаєте файл напряму)
def main():
    st.title("🌊 Хвильовий Пакет")
    st.sidebar.header("Налаштування")
    from scipy import constants
    
    params = {}
    params['m'] = constants.m_e
    params['E'] = st.sidebar.number_input("Енергія E (еВ)", value=5.0, step=0.1, key="wp_E") * EV
    params['U0'] = st.sidebar.number_input("Висота U₀ (еВ)", value=10.0, step=0.1, key="wp_U0") * EV
    
    run_wave_packet_simulation(params)

if __name__ == "__main__":
    main()