import numpy as np
import pandas as pd
from scipy.optimize import linprog
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ============================================================
# INCISO C
# Interpretación aplicada:
# El techo de 650 horas sustituye la cuota mínima anterior
# de 800 horas para el Consultor Principal.
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

    # Se elimina:
    # ([0, 1, 2, 3], 800)
    # porque se interpreta sustituida por Consultor Principal <= 650

    ([4, 5, 6, 7], 400),    # Ingeniero Especialista >= 400
    ([8, 9, 10, 11], 300)   # Analista de Soporte >= 300
]

for indices, valor in restricciones_ge:
    fila = np.zeros(12)
    fila[indices] = 1
    A_ub.append(-fila)
    b_ub.append(-valor)

# ============================================================
# RESTRICCIONES NUEVAS DEL INCISO C
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

resultado_C = linprog(
    c=costos,
    A_ub=np.array(A_ub),
    b_ub=np.array(b_ub),
    A_eq=np.array(A_eq),
    b_eq=np.array(b_eq),
    bounds=bounds,
    method="highs"
)

if not resultado_C.success:
    raise ValueError("El modelo no encontró una solución óptima factible.")

# ============================================================
# TABLAS DE RESULTADOS
# ============================================================

solucion_C = resultado_C.x.reshape(3, 4)
matriz_costos = costos.reshape(3, 4)

df_horas_C = pd.DataFrame(
    solucion_C,
    index=perfiles,
    columns=bloques
)

df_costos_unitarios_C = pd.DataFrame(
    matriz_costos,
    index=perfiles,
    columns=bloques
)

df_costos_totales_C = df_horas_C * df_costos_unitarios_C

df_resultado_C = df_horas_C.copy()
df_resultado_C["Total por perfil"] = df_resultado_C.sum(axis=1)

total_por_bloque_C = df_horas_C.sum(axis=0)
total_por_perfil_C = df_horas_C.sum(axis=1)

costo_operativo_C = resultado_C.fun
cotizacion_C = costo_operativo_C * 1.15

print("SOLUCIÓN ÓPTIMA - INCISO C")
print(df_resultado_C)

print("\nTOTAL POR BLOQUE")
print(total_por_bloque_C)

print("\nCOSTO OPERATIVO C:")
print(f"${costo_operativo_C:,.2f}")

print("\nCOTIZACIÓN C CON 15% DE MARGEN:")
print(f"${cotizacion_C:,.2f}")



# ============================================================
# PREPARACIÓN DE DATOS PARA GRÁFICOS
# ============================================================

df_costo_variable_C = pd.DataFrame({
    "Variable": variables,
    "Costo total": df_costos_totales_C.values.flatten()
})

# ============================================================
# DASHBOARD PLOTLY GRAPH OBJECTS
# ============================================================

fig_C = make_subplots(
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

# Gráfico 1: Horas por perfil
fig_C.add_trace(
    go.Bar(
        x=total_por_perfil_C.index,
        y=total_por_perfil_C.values,
        text=[f"{v:,.0f}" for v in total_por_perfil_C.values],
        textposition="outside",
        name="Horas por perfil",
        hovertemplate="<b>%{x}</b><br>Horas asignadas: %{y:,.0f}<extra></extra>"
    ),
    row=1,
    col=1
)

# Gráfico 2: Horas por bloque
fig_C.add_trace(
    go.Bar(
        x=total_por_bloque_C.index,
        y=total_por_bloque_C.values,
        text=[f"{v:,.0f}" for v in total_por_bloque_C.values],
        textposition="outside",
        name="Horas por bloque",
        hovertemplate="<b>%{x}</b><br>Horas asignadas: %{y:,.0f}<extra></extra>"
    ),
    row=1,
    col=2
)

# Gráfico 3: Heatmap Xij
fig_C.add_trace(
    go.Heatmap(
        z=df_horas_C.values,
        x=bloques,
        y=perfiles,
        text=df_horas_C.values,
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

# Gráfico 4: Costo por variable
fig_C.add_trace(
    go.Bar(
        x=df_costo_variable_C["Variable"],
        y=df_costo_variable_C["Costo total"],
        text=[f"${v:,.0f}" for v in df_costo_variable_C["Costo total"]],
        textposition="outside",
        name="Costo por variable",
        hovertemplate="<b>%{x}</b><br>Costo: $%{y:,.2f}<extra></extra>"
    ),
    row=2,
    col=2
)

fig_C.update_layout(
    title=(
        f"Inciso C: solución óptima | "
        f"Costo operativo: ${costo_operativo_C:,.2f} | "
        f"Cotización: ${cotizacion_C:,.2f}"
    ),
    template="plotly_white",
    width=1350,
    height=900,
    showlegend=False
)

fig_C.update_yaxes(title_text="Horas", row=1, col=1)
fig_C.update_yaxes(title_text="Horas", row=1, col=2)
fig_C.update_yaxes(title_text="Perfil profesional", row=2, col=1)
fig_C.update_yaxes(title_text="Costo en dólares", row=2, col=2)

fig_C.update_xaxes(title_text="Perfil profesional", row=1, col=1)
fig_C.update_xaxes(title_text="Bloque técnico", row=1, col=2)
fig_C.update_xaxes(title_text="Bloque técnico", row=2, col=1)
fig_C.update_xaxes(title_text="Variable de decisión", row=2, col=2)

# ============================================================
# EXPORTACIÓN HTML CON PLOTLYJS INCLUIDO
# ============================================================

archivo_html = Path("inciso_C_dashboard.html").resolve()

fig_C.write_html(
    str(archivo_html),
    include_plotlyjs=True,
    full_html=True,
    auto_open=False
)

print(f"Dashboard generado correctamente en:")
print(archivo_html)

# Mostrar en notebook o entorno compatible
fig_C.show()
