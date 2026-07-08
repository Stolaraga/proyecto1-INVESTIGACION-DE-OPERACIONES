import numpy as np
import pandas as pd
from scipy.optimize import linprog
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ============================================================
# INCISO D
# Se mantienen las restricciones del inciso C corregido.
# Se actualizan únicamente los costos del bloque j=1.
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
costos_D = np.array([
    6.50, 5.75, 6.25, 5.00,
    6.75, 6.50, 6.50, 6.25,
    7.00, 7.50, 7.50, 7.25
])

# ============================================================
# RESTRICCIÓN TOTAL DE HORAS
# ============================================================

A_eq = [np.ones(12)]
b_eq = [2000]

A_ub = []
b_ub = []

# ============================================================
# RESTRICCIONES >= CONVERTIDAS A <=
# ============================================================

restricciones_ge = [
    ([0, 4, 8], 400),    # Escaneos >= 400
    ([1, 5, 9], 600),    # Pentesting >= 600
    ([2, 6, 10], 300),   # Cifrado >= 300
    ([3, 7, 11], 300),   # Monitoreo >= 300

    # Se mantiene interpretación del inciso C:
    # Consultor Principal >= 800 se sustituye por Consultor Principal <= 650

    ([4, 5, 6, 7], 400),    # Ingeniero Especialista >= 400
    ([8, 9, 10, 11], 300)   # Analista de Soporte >= 300
]

for indices, valor in restricciones_ge:
    fila = np.zeros(12)
    fila[indices] = 1
    A_ub.append(-fila)
    b_ub.append(-valor)

# ============================================================
# TECHOS DEL INCISO C
# ============================================================

# Escaneos <= 600
fila = np.zeros(12)
fila[[0, 4, 8]] = 1
A_ub.append(fila)
b_ub.append(600)

# Consultor Principal <= 650
fila = np.zeros(12)
fila[[0, 1, 2, 3]] = 1
A_ub.append(fila)
b_ub.append(650)

# ============================================================
# RESTRICCIÓN DEL INCISO B
# Cada combinación perfil-bloque debe tener al menos 50 horas
# ============================================================

bounds = [(50, None)] * 12

# ============================================================
# RESOLUCIÓN DEL MODELO
# ============================================================

resultado_D = linprog(
    c=costos_D,
    A_ub=np.array(A_ub),
    b_ub=np.array(b_ub),
    A_eq=np.array(A_eq),
    b_eq=np.array(b_eq),
    bounds=bounds,
    method="highs"
)

if not resultado_D.success:
    raise ValueError("El modelo no encontró una solución óptima factible.")

# ============================================================
# TABLAS DE RESULTADOS
# ============================================================

solucion_D = resultado_D.x.reshape(3, 4)
matriz_costos_D = costos_D.reshape(3, 4)

df_horas_D = pd.DataFrame(
    solucion_D,
    index=perfiles,
    columns=bloques
)

df_costos_unitarios_D = pd.DataFrame(
    matriz_costos_D,
    index=perfiles,
    columns=bloques
)

df_costos_totales_D = df_horas_D * df_costos_unitarios_D

df_resultado_D = df_horas_D.copy()
df_resultado_D["Total por perfil"] = df_resultado_D.sum(axis=1)

total_por_bloque_D = df_horas_D.sum(axis=0)
total_por_perfil_D = df_horas_D.sum(axis=1)

costo_operativo_D = resultado_D.fun
cotizacion_D = costo_operativo_D * 1.15

print("SOLUCIÓN ÓPTIMA - INCISO D")
print(df_resultado_D)

print("\nTOTAL POR BLOQUE")
print(total_por_bloque_D)

print("\nCOSTO OPERATIVO D:")
print(f"${costo_operativo_D:,.2f}")

print("\nCOTIZACIÓN D CON 15% DE MARGEN:")
print(f"${cotizacion_D:,.2f}")

# ============================================================
# DASHBOARD CON PLOTLY GRAPH OBJECTS
# ============================================================

df_costo_variable_D = pd.DataFrame({
    "Variable": variables,
    "Costo total": df_costos_totales_D.values.flatten()
})

fig_D = make_subplots(
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

fig_D.add_trace(
    go.Bar(
        x=total_por_perfil_D.index,
        y=total_por_perfil_D.values,
        text=[f"{v:,.0f}" for v in total_por_perfil_D.values],
        textposition="outside",
        name="Horas por perfil",
        hovertemplate="<b>%{x}</b><br>Horas asignadas: %{y:,.0f}<extra></extra>"
    ),
    row=1,
    col=1
)

fig_D.add_trace(
    go.Bar(
        x=total_por_bloque_D.index,
        y=total_por_bloque_D.values,
        text=[f"{v:,.0f}" for v in total_por_bloque_D.values],
        textposition="outside",
        name="Horas por bloque",
        hovertemplate="<b>%{x}</b><br>Horas asignadas: %{y:,.0f}<extra></extra>"
    ),
    row=1,
    col=2
)

fig_D.add_trace(
    go.Heatmap(
        z=df_horas_D.values,
        x=bloques,
        y=perfiles,
        text=df_horas_D.values,
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

fig_D.add_trace(
    go.Bar(
        x=df_costo_variable_D["Variable"],
        y=df_costo_variable_D["Costo total"],
        text=[f"${v:,.0f}" for v in df_costo_variable_D["Costo total"]],
        textposition="outside",
        name="Costo por variable",
        hovertemplate="<b>%{x}</b><br>Costo: $%{y:,.2f}<extra></extra>"
    ),
    row=2,
    col=2
)

fig_D.update_layout(
    title=(
        f"Inciso D: solución óptima | "
        f"Costo operativo: ${costo_operativo_D:,.2f} | "
        f"Cotización: ${cotizacion_D:,.2f}"
    ),
    template="plotly_white",
    width=1350,
    height=900,
    showlegend=False
)

fig_D.update_yaxes(title_text="Horas", row=1, col=1)
fig_D.update_yaxes(title_text="Horas", row=1, col=2)
fig_D.update_yaxes(title_text="Perfil profesional", row=2, col=1)
fig_D.update_yaxes(title_text="Costo en dólares", row=2, col=2)

fig_D.update_xaxes(title_text="Perfil profesional", row=1, col=1)
fig_D.update_xaxes(title_text="Bloque técnico", row=1, col=2)
fig_D.update_xaxes(title_text="Bloque técnico", row=2, col=1)
fig_D.update_xaxes(title_text="Variable de decisión", row=2, col=2)

archivo_html = Path("inciso_D_dashboard.html").resolve()

fig_D.write_html(
    str(archivo_html),
    include_plotlyjs=True,
    full_html=True,
    auto_open=False
)

print("\nDashboard generado correctamente en:")
print(archivo_html)

fig_D.show()
