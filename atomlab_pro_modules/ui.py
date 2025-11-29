
# ui.py - helper UI functions for Streamlit rendering
import streamlit as st
import plotly.graph_objects as go
import numpy as np

def plot_bohr_svg(n_list, angle=0.0, highlight=None, size=500):
    # returns html svg string
    base = 40
    cx = cy = size//2
    svg = [f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' xmlns='http://www.w3.org/2000/svg'>"]
    svg.append(f"<rect width='100%' height='100%' fill='#07070a' />")
    svg.append(f"<circle cx='{cx}' cy='{cy}' r='8' fill='#ff6666' />")
    for n in n_list:
        r = base * n
        svg.append(f"<circle cx='{cx}' cy='{cy}' r='{r}' stroke='#4c8cff' fill='none' stroke-width='2' opacity='0.6' />")
    if highlight is not None:
        r = base * highlight
        ex = cx + r * math.cos(angle)
        ey = cy + r * math.sin(angle)
        svg.append(f"<circle cx='{ex}' cy='{ey}' r='6' fill='#00ffcc' />")
    svg.append("</svg>")
    return '\n'.join(svg)

def plot_orbital_plotly(X,Y,Z,Values, isosurface=0.02):
    fig = go.Figure(data=go.Isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=Values.flatten(), isomin=Values.max()*isosurface, isomax=Values.max(), surface_count=4))
    fig.update_layout(scene=dict(xaxis_visible=False,yaxis_visible=False,zaxis_visible=False), margin=dict(t=0,l=0,r=0,b=0))
    return fig
