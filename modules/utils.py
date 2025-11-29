# modules/utils.py
# Цей файл містить константи та допоміжні функції для квантових симуляцій

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants

# ──────────────────────── ГОЛОВНІ ФІЗИЧНІ КОНСТАНТИ ────────────────────────
# Вони доступні для імпорту з будь-якого файлу: from modules.utils import HBAR, M_E, EV

HBAR = constants.hbar           # приведена стала Планка (Дж·с)
M_E = constants.m_e             # маса електрона (кг)
M_P = constants.m_p             # маса протона (кг)
EV = constants.electron_volt    # 1 еВ у джоулях (Дж)

# ──────────────────────── ДОПОМІЖНІ ФУНКЦІЇ ────────────────────────

def get_k(E, m, U=0.0):
    """
    Повертає хвильове число k = sqrt(2m(E-U))/ℏ
    Якщо E < U — повертає уявне число (експоненційне затухання)
    """
    if m <= 0:
        return 0.0
    val = 2.0 * m * (E - U)
    return np.sqrt(val + 0j) / HBAR  # +0j — щоб завжди був комплексний тип


def draw_arrow(ax, x1, x2, y, text, color='white'):
    """Малює двосторонню стрілку з підписом (наприклад, довжина ями)."""
    if abs(x2 - x1) < 1e-20:
        return
    ax.annotate('', xy=(x1, y), xytext=(x2, y),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.5))
    ax.text((x1 + x2) / 2, y, text,
            ha='center', va='bottom', color=color, fontsize=10,
            bbox=dict(facecolor='#0E1117', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))


def plot_setup(ax, title, xlabel="x (нм)", ylabel="Енергія (еВ)"):
    """Темна тема графіка, як у Streamlit."""
    ax.set_title(title, color='white', fontsize=16, pad=15)
    ax.set_xlabel(xlabel, color='white', fontsize=12)
    ax.set_ylabel(ylabel, color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.set_facecolor('#0E1117')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')