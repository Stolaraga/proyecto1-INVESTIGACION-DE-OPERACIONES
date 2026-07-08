import numpy as np
import pandas as pd
from scipy.optimize import linprog
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import webbrowser

# ============================================================
# FUNCIÓN PARA GUARDAR Y ABRIR GRÁFICOS SIN USAR 127.0.0.1
# ============================================================

def guardar_y_abrir(fig, nombre_archivo):
    ruta = Path(nombre_archivo).resolve()
    
    fig.write_html(
        str(ruta),
        include_plotlyjs=True,
        full_html=True,
        auto_open=False
    )
    
    webbrowser.open(ruta.as_uri())
    print(f"Archivo generado: {ruta}")

# ============================================================
# ESCENARIO A: MODELO BASE
# ============================================================

variables = [
    "X11", "X12", "X13", "X14",
    "X21", "X22", "X23", "X24",
    "X31", "X32", "X33", "X34"
]

perfiles = [
    "Consultor Principal / QSA",
    "Ingeniero Especialista",
    "Analista de Soporte"
]

bloques = [
    "Escaneos",
    "Pentesting",
    "Cifrado",
    "Monitoreo"
]

costos = np.array([
    4.75, 5.75, 6.25, 5.00,
    5.25, 6.50, 6.50, 6.25,
    6.50, 7.50, 7.50, 7.25
])

# Restricción total de horas
A_eq = [np.ones(12)]
b_eq = [2000]

# Restricciones >= convertidas a <=
restricciones_ge = []
valores_ge = []

# Escaneos >= 400
fila = np.zeros(12)
fila[[0, 4, 8]] = 1
restricciones_ge.append(fila)
valores_ge.append(400)

# Pentesting >= 600
fila = np.zeros(12)
fila[[1, 5, 9]] = 1
restricciones_ge.append(fila)
valores_ge.append(600)

# Cifrado >= 300
fila = np.zeros(12)
fila[[2, 6, 10]] = 1
restricciones_ge.append(fila)
valores_ge.append(300)

# Monitoreo >= 300
fila = np.zeros(12)
fila[[3, 7, 11]] = 1
restricciones_ge.append(fila)
valores_ge.append(300)

# Consultor Principal >= 800
fila = np.zeros(12)
fila[[0, 1, 2, 3]] = 1
restricciones_ge.append(fila)
valores_ge.append(800)

# Ingeniero Especialista >= 400
fila = np.zeros(12)
fila[[4, 5, 6, 7]] = 1
restricciones_ge.append(fila)
valores_ge.append(400)

# Analista de Soporte >= 300
fila = np.zeros(12)
fila[[8, 9, 10, 11]] = 1
restricciones_ge.append(fila)
valores_ge.append(300)

A_ub = -np.array(restricciones_ge)
b_ub = -np.array(valores_ge)

bounds = [(0, None)] * 12

resultado = linprog(
    c=costos,
    A_ub=A_ub,
    b_ub=b_ub,
    A_eq=A_eq,
    b_eq=b_eq,
    bounds=bounds,
    method="highs"
)

if not resultado.success:
    raise ValueError("El modelo no encontró una solución óptima factible.")

# ============================================================
# TABLAS
# ============================================================

solucion = resultado.x.reshape(3, 4)
matriz_costos = costos.reshape(3, 4)

df_horas = pd.DataFrame(
    solucion,
    index=perfiles,
    columns=bloques
)

df_costos_unitarios = pd.DataFrame(
    matriz_costos,
    index=perfiles,
    columns=bloques
)

df_costos_totales = df_horas * df_costos_unitarios

total_por_perfil = df_horas.sum(axis=1)
total_por_bloque = df_horas.sum(axis=0)

costo_operativo = resultado.fun
cotizacion = costo_operativo * 1.15

print("SOLUCIÓN ÓPTIMA - ESCENARIO A")
print(df_horas)

print("\nCOSTO OPERATIVO MÍNIMO:")
print(f"${costo_operativo:,.2f}")

print("\nCOTIZACIÓN CON 15% DE MARGEN:")
print(f"${cotizacion:,.2f}")

# ============================================================
# DASHBOARD PLOTLY GRAPH OBJECTS
# ============================================================

df_costo_variable = pd.DataFrame({
    "Variable": variables,
    "Costo total": df_costos_totales.values.flatten()
})

fig_dashboard = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        "Horas por perfil profesional",
        "Horas por bloque técnico",
        "Matriz de asignación Xij",
        "Costo por variable"
    ],
    specs=[
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "heatmap"}, {"type": "bar"}]
    ]
)

fig_dashboard.add_trace(
    go.Bar(
        x=total_por_perfil.index,
        y=total_por_perfil.values,
        text=[f"{v:,.0f}" for v in total_por_perfil.values],
        textposition="outside",
        name="Horas por perfil",
        hovertemplate="<b>%{x}</b><br>Horas: %{y:,.0f}<extra></extra>"
    ),
    row=1,
    col=1
)

fig_dashboard.add_trace(
    go.Bar(
        x=total_por_bloque.index,
        y=total_por_bloque.values,
        text=[f"{v:,.0f}" for v in total_por_bloque.values],
        textposition="outside",
        name="Horas por bloque",
        hovertemplate="<b>%{x}</b><br>Horas: %{y:,.0f}<extra></extra>"
    ),
    row=1,
    col=2
)

fig_dashboard.add_trace(
    go.Heatmap(
        z=df_horas.values,
        x=bloques,
        y=perfiles,
        text=df_horas.values,
        texttemplate="%{text:.0f}",
        colorbar=dict(title="Horas"),
        name="Matriz Xij",
        hovertemplate=(
            "<b>Perfil:</b> %{y}<br>"
            "<b>Bloque:</b> %{x}<br>"
            "<b>Horas:</b> %{z:,.0f}<extra></extra>"
        )
    ),
    row=2,
    col=1
)

fig_dashboard.add_trace(
    go.Bar(
        x=df_costo_variable["Variable"],
        y=df_costo_variable["Costo total"],
        text=[f"${v:,.0f}" for v in df_costo_variable["Costo total"]],
        textposition="outside",
        name="Costo por variable",
        hovertemplate="<b>%{x}</b><br>Costo: $%{y:,.2f}<extra></extra>"
    ),
    row=2,
    col=2
)

fig_dashboard.update_layout(
    title=(
        f"Escenario A: solución óptima | "
        f"Costo operativo: ${costo_operativo:,.2f} | "
        f"Cotización: ${cotizacion:,.2f}"
    ),
    template="plotly_white",
    width=1300,
    height=850,
    showlegend=False
)

fig_dashboard.update_yaxes(title_text="Horas", row=1, col=1)
fig_dashboard.update_yaxes(title_text="Horas", row=1, col=2)
fig_dashboard.update_yaxes(title_text="Costo en dólares", row=2, col=2)

# Guardar y abrir sin servidor local
guardar_y_abrir(fig_dashboard, "escenario_A_dashboard.html")
