import matplotlib.pyplot as plt
import cmath
from scipy import constants
from scipy.integrate import simps


# --- КОНСТАНТИ ---
HBAR = constants.hbar
M_E = constants.m_e
EV = constants.electron_volt

# --- ФУНКЦІЇ ---

def get_k(E, m, U=0):
    """Розраховує хвильовий вектор k."""
    if m <= 0: return 0.0
    val = 2 * m * (E - U)
    return cmath.sqrt(val) / HBAR

def plot_setup(ax, title, U_max=None):
    """Базове налаштування графіків (темна тема)."""
    ax.set_title(title, color='white', fontsize=14)
    ax.set_xlabel("x (м)", color='white')
    ax.set_ylabel("Енергія / Ψ", color='white')
    
    # Налаштування кольорів для темної теми
    ax.set_facecolor('#0e1117')
    fig = ax.figure
    fig.patch.set_facecolor('#0e1117')
    
    ax.tick_params(colors='white')
    for spine in ['left', 'bottom', 'right', 'top']:
        ax.spines[spine].set_color('white')
        
    ax.grid(True, linestyle=':', alpha=0.3, color='gray')

def draw_arrow(ax, x1, x2, y, text, color='white'):
    """Малює стрілку розміру."""
    if abs(x2 - x1) < 1e-20: return
    ax.annotate('', xy=(x1, y), xytext=(x2, y), arrowprops=dict(arrowstyle='<->', color=color))
    ax.text((x1 + x2) / 2.0, y, text, ha='center', va='bottom', color=color,
            bbox=dict(facecolor='#0e1117', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.1'))