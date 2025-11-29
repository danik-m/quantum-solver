try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps

__all__ = ["simps"]