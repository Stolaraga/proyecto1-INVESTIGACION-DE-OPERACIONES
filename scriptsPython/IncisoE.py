import numpy as np
import pandas as pd
from scipy.optimize import linprog
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ============================================================
# INCISO E
# Lectura aplicada:
# Se mantienen los costos actualizados del inciso D.
# Las cuotas mínimas se transforman en igualdades exactas.
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

# Costos actualizados del inciso D
costos_E = np.array([
    6.50, 5.75, 6.25, 5.00,
    6.75, 6.50, 6.50, 6.25,
    7.00, 7.50, 7.50, 7.25
])

# ============================================================
# RESTRICCIONES DE IGUALDAD
# ============================================================

A_eq = []
b_eq = []

# Por bloque técnico:
# Escaneos = 500
fila = np.zeros(12)
fila[[0, 4, 8]] = 1
A_eq.append(fila)
b_eq.append(500)

# Pentesting = 700
fila = np.zeros(12)
fila[[1, 5, 9]] = 1
A_eq.append(fila)
b_eq.append(700)

# Cifrado = 400
fila = np.zeros(12)
fila[[2, 6, 10]] = 1
A_eq.append(fila)
b_eq.append(400)

# Monitoreo = 400
fila = np.zeros(12)
fila[[3, 7, 11]] = 1
A_eq.append(fila)
b_eq.append(400)

# Por perfil profesional:
# Consultor Principal = 400
fila = np.zeros(12)
fila[[0, 1, 2, 3]] = 1
A_eq.append(fila)
b_eq.append(400)

# Ingeniero Especialista = 1000
fila = np.zeros(12)
fila[[4, 5, 6, 7]] = 1
A_eq.append(fila)
b_eq.append(1000)

# Analista de Soporte = 600
fila = np.zeros(12)
fila[[8, 9, 10, 11]] = 1
A_eq.append(fila)
b_eq.append(600)

# No negatividad
bounds = [(0, None)] * 12

# ============================================================
# RESOLUCIÓN DEL MODELO
# ============================================================

resultado_E = linprog(
    c=costos_E,
    A_eq=np.array(A_eq),
    b_eq=np.array(b_eq),
    bounds=bounds,
    method="highs"
)

if not resultado_E.success:
    raise ValueError("El modelo no encontró una solución óptima factible.")

# ============================================================
# TABLAS DE RESULTADOS
# ============================================================

solucion_E = resultado_E.x.reshape(3, 4)
matriz_costos_E = costos_E.reshape(3, 4)

df_horas_E = pd.DataFrame(
    solucion_E,
    index=perfiles,
    columns=bloques
)

df_costos_unitarios_E = pd.DataFrame(
    matriz_costos_E,
    index=perfiles,
    columns=bloques
)

df_costos_totales_E = df_horas_E * df_costos_unitarios_E

df_resultado_E = df_horas_E.copy()
df_resultado_E["Total por perfil"] = df_resultado_E.sum(axis=1)

total_por_bloque_E = df_horas_E.sum(axis=0)
total_por_perfil_E = df_horas_E.sum(axis=1)

costo_operativo_E = resultado_E.fun
cotizacion_E = costo_operativo_E * 1.15

print("SOLUCIÓN ÓPTIMA - INCISO E")
print(df_resultado_E)

print("\nTOTAL POR BLOQUE")
print(total_por_bloque_E)

print("\nCOSTO OPERATIVO E:")
print(f"${costo_operativo_E:,.2f}")

print("\nCOTIZACIÓN E CON 15% DE MARGEN:")
print(f"${cotizacion_E:,.2f}")

# ============================================================
# DASHBOARD CON PLOTLY GRAPH OBJECTS
# ============================================================

df_costo_variable_E = pd.DataFrame({
    "Variable": variables,
    "Costo total": df_costos_totales_E.values.flatten()
})

fig_E = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        "Horas por perfil profesional",
        "Horas por bloque técnico",
        "Matriz de asignación Xij",
        "Costo operativo por variable"
    ],
    specs=[
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "heatmap"}, {"type": "bar"}]
    ]
)

fig_E.add_trace(
    go.Bar(
        x=total_por_perfil_E.index,
        y=total_por_perfil_E.values,
        text=[f"{v:,.0f}" for v in total_por_perfil_E.values],
        textposition="outside",
        name="Horas por perfil",
        hovertemplate="<b>%{x}</b><br>Horas asignadas: %{y:,.0f}<extra></extra>"
    ),
    row=1,
    col=1
)

fig_E.add_trace(
    go.Bar(
        x=total_por_bloque_E.index,
        y=total_por_bloque_E.values,
        text=[f"{v:,.0f}" for v in total_por_bloque_E.values],
        textposition="outside",
        name="Horas por bloque",
        hovertemplate="<b>%{x}</b><br>Horas asignadas: %{y:,.0f}<extra></extra>"
    ),
    row=1,
    col=2
)

fig_E.add_trace(
    go.Heatmap(
        z=df_horas_E.values,
        x=bloques,
        y=perfiles,
        text=df_horas_E.values,
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

fig_E.add_trace(
    go.Bar(
        x=df_costo_variable_E["Variable"],
        y=df_costo_variable_E["Costo total"],
        text=[f"${v:,.0f}" for v in df_costo_variable_E["Costo total"]],
        textposition="outside",
        name="Costo por variable",
        hovertemplate="<b>%{x}</b><br>Costo: $%{y:,.2f}<extra></extra>"
    ),
    row=2,
    col=2
)

fig_E.update_layout(
    title=(
        f"Inciso E: solución óptima | "
        f"Costo operativo: ${costo_operativo_E:,.2f} | "
        f"Cotización: ${cotizacion_E:,.2f}"
    ),
    template="plotly_white",
    width=1350,
    height=900,
    showlegend=False
)

fig_E.update_yaxes(title_text="Horas", row=1, col=1)
fig_E.update_yaxes(title_text="Horas", row=1, col=2)
fig_E.update_yaxes(title_text="Perfil profesional", row=2, col=1)
fig_E.update_yaxes(title_text="Costo en dólares", row=2, col=2)

fig_E.update_xaxes(title_text="Perfil profesional", row=1, col=1)
fig_E.update_xaxes(title_text="Bloque técnico", row=1, col=2)
fig_E.update_xaxes(title_text="Bloque técnico", row=2, col=1)
fig_E.update_xaxes(title_text="Variable de decisión", row=2, col=2)

archivo_html = Path("inciso_E_dashboard.html").resolve()

fig_E.write_html(
    str(archivo_html),
    include_plotlyjs=True,
    full_html=True,
    auto_open=False
)

print("\nDashboard generado correctamente en:")
print(archivo_html)

fig_E.show()
