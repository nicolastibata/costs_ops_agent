"""
agent.py — Agente conversacional para gestión de costos operativos.

Herramientas:
  - get_projection        : consulta pronósticos del modelo ECM
  - get_model_summary     : métricas y metodología del modelo
  - compare_scenarios     : escenario optimista / base / pesimista
  - plot_projection       : genera y guarda gráfico de proyección
  - web_search            : busca noticias de mercado via Anthropic web_search tool
  - explain_concept       : explica ECM, agente de IA, cointegración, etc.

Uso:
  export ANTHROPIC_API_KEY="..."
  python agent.py
"""

import json
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # sin GUI — funciona en terminal
import matplotlib.pyplot as plt
from anthropic import Anthropic

client = Anthropic()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_ROOT, "docs", "outputs", "proyeccion_equipos.csv")
PLOT_DIR = os.path.join(_ROOT, "docs", "outputs")

# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN DE HERRAMIENTAS
# ══════════════════════════════════════════════════════════════════════════════

def get_projection(equipo: str) -> str:
    """Retorna la proyección mensual para un equipo."""
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    except FileNotFoundError:
        return "ERROR: archivo de proyección no encontrado. Ejecuta primero el notebook."

    mask = df["equipo"].str.lower().str.contains(equipo.lower())
    subset = df[mask]
    if subset.empty:
        return f"No se encontraron proyecciones para '{equipo}'. Opciones: Price_Equipo1, Price_Equipo2."

    rows = []
    for _, r in subset.iterrows():
        rows.append({
            "mes":       r["Date"].strftime("%Y-%m"),
            "pred_mean": round(r["pred_mean"], 2),
            "lower_80":  round(r["lower_80"],  2),
            "upper_80":  round(r["upper_80"],  2),
            "lower_95":  round(r["lower_95"],  2),
            "upper_95":  round(r["upper_95"],  2),
        })
    return json.dumps(rows, ensure_ascii=False, indent=2)


def get_model_summary() -> str:
    """Retorna métricas y metodología del modelo."""
    summary = {
        "modelos": {
            "Price_Equipo1": {
                "driver":                "Price_Y",
                "tipo":                  "Error Correction Model (ECM)",
                "R2_largo_plazo":        0.9932,
                "R2_ECM":                0.4770,
                "MAPE_test":             "1.72%",
                "MAE_test":              10.17,
                "RMSE_test":             163.92,
                "R2_test":               0.9866,
                "variables_descartadas": [
                    "Price_X (sin relación)",
                    "Price_Z (p=0.672 con Y presente)"
                ],
            },
            "Price_Equipo2": {
                "driver":                "Price_Z",
                "tipo":                  "Error Correction Model (ECM)",
                "R2_largo_plazo":        0.9656,
                "R2_ECM":                0.2904,
                "MAPE_test":             "3.24%",
                "MAE_test":              36.20,
                "RMSE_test":             1978.45,
                "R2_test":               0.9220,
                "variables_descartadas": [
                    "Price_X (sin relación)",
                    "Price_Y (redundante con Z presente)"
                ],
            },
        },
        "metodologia": {
            "seleccion_variables":  "Correlación con rezagos + ADF + cointegración Engle-Granger + Lasso",
            "modelo_mp":            "Holt-Winters (tendencia aditiva + estacionalidad 12 meses)",
            "intervalos":           "Bootstrap de residuales ECM (1000 simulaciones), percentiles 2.5/10/90/97.5",
            "horizonte":            "12 meses",
            "split":                "80% train (2010-2020) / 20% test (2020-2023), temporal sin shuffle",
        },
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def compare_scenarios(equipo: str) -> str:
    """
    Genera escenarios optimista / base / pesimista usando los IC del modelo.
    Optimista  = lower_80  (materia prima en zona baja)
    Base       = pred_mean
    Pesimista  = upper_80  (materia prima en zona alta)
    """
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    except FileNotFoundError:
        return "ERROR: archivo de proyección no encontrado."

    mask = df["equipo"].str.lower().str.contains(equipo.lower())
    subset = df[mask]
    if subset.empty:
        return f"No se encontraron datos para '{equipo}'."

    scenarios = []
    for _, r in subset.iterrows():
        scenarios.append({
            "mes":       r["Date"].strftime("%Y-%m"),
            "optimista": round(r["lower_80"],  2),
            "base":      round(r["pred_mean"], 2),
            "pesimista": round(r["upper_80"],  2),
            "rango_95":  f"{round(r['lower_95'], 1)} – {round(r['upper_95'], 1)}",
        })

    return json.dumps({
        "equipo":    equipo,
        "escenarios": scenarios,
        "nota": (
            "Optimista = p10 bootstrap (MP en zona baja). "
            "Pesimista = p90 (MP en zona alta). "
            "Rango 95% = intervalo completo del modelo."
        ),
    }, ensure_ascii=False, indent=2)


def plot_projection(equipo: str) -> str:
    """Genera y guarda un PNG con proyección e intervalos de confianza."""
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    except FileNotFoundError:
        return "ERROR: archivo de proyección no encontrado."

    mask = df["equipo"].str.lower().str.contains(equipo.lower())
    subset = df[mask]
    if subset.empty:
        return f"No hay datos para '{equipo}'."

    dates = subset["Date"]
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.fill_between(dates, subset["lower_95"], subset["upper_95"],
                    alpha=0.12, color="#378ADD", label="IC 95%")
    ax.fill_between(dates, subset["lower_80"], subset["upper_80"],
                    alpha=0.25, color="#378ADD", label="IC 80%")
    ax.plot(dates, subset["pred_mean"],
            color="#378ADD", linewidth=2.5, marker="o", markersize=4, label="Pronóstico")

    ax.set_title(f"Proyección 12 meses — {equipo}", fontsize=13)
    ax.set_ylabel("Costo estimado")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    slug = equipo.lower().replace(" ", "_")
    path = os.path.join(PLOT_DIR, f"plot_{slug}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    subprocess.Popen(["open", path])   # abre el PNG con el visor del sistema (macOS)
    return f"Gráfico guardado y abierto: {path}"


def explain_concept(concepto: str) -> str:
    """Explica conceptos técnicos: 'ecm', 'agente', 'cointegracion'."""
    conceptos = {
        "ecm": {
            "nombre": "Error Correction Model (ECM)",
            "intuicion": (
                "El ECM captura dos dinámicas simultáneas: "
                "(1) Corto plazo: cómo cambia el precio del equipo hoy en respuesta a cambios en la materia prima. "
                "(2) Corrección: si el precio se aleja del equilibrio de largo plazo, una fuerza lo jala de vuelta."
            ),
            "ecuacion": "Δy_t = α + β·Δx_t + γ·(y_{t-1} - λ·x_{t-1}) + ε_t",
            "terminos": {
                "β":  "Efecto de corto plazo (transmisión inmediata del cambio en MP al equipo)",
                "γ":  "Velocidad de corrección al equilibrio (negativo, entre -1 y 0)",
                "EC": "Término de corrección = desequilibrio del período anterior",
            },
            "por_que_no_ols": (
                "OLS en series no estacionarias produce regresiones espurias: "
                "correlaciones altas aunque no haya relación real. "
                "El ADF confirmó que X, Y y Equipo1 no son estacionarias."
            ),
            "resultados_del_proyecto": {
                "Equipo1~Y": "γ=-0.77 → corrige 77% del desequilibrio por día (relación muy fuerte y rápida)",
                "Equipo2~Z": "γ=-0.26 → corrige 26% por día (relación sólida pero más gradual)",
            },
        },
        "agente": {
            "nombre": "IA Convencional vs Agente de IA",
            "ia_convencional": {
                "definicion": "Sistema que recibe un input y produce un output (predicción, texto, clasificación).",
                "limitaciones": [
                    "Reactivo: responde a un input puntual, no planifica",
                    "Sin memoria entre llamadas",
                    "Sin acceso a herramientas externas",
                    "No puede encadenar acciones",
                ],
                "ejemplo": "El modelo ECM: dado Price_Y de hoy → predice costo de Equipo1.",
            },
            "agente_ia": {
                "definicion": (
                    "Sistema autónomo que percibe su entorno, decide qué acciones tomar "
                    "y las ejecuta iterativamente hasta alcanzar un objetivo."
                ),
                "pilares": {
                    "Autonomía":     "Decide qué herramienta usar y cuándo, sin instrucción paso a paso.",
                    "Herramientas":  "Accede a APIs, archivos, búsqueda web, generación de gráficos.",
                    "Memoria":       "Mantiene historial de la conversación para dar respuestas coherentes.",
                    "Acción":        "Ejecuta código, escribe archivos, llama servicios externos.",
                    "Planificación": "Encadena múltiples herramientas para resolver objetivos complejos.",
                },
                "ejemplo": (
                    "Este agente: ante '¿cómo están las perspectivas del Equipo 2?' decide autónomamente: "
                    "1) get_projection(Equipo2), 2) web_search('zinc price outlook'), "
                    "3) compare_scenarios(Equipo2), 4) sintetiza todo en una respuesta."
                ),
            },
            "este_agente": {
                "herramientas":  ["get_projection", "get_model_summary", "compare_scenarios",
                                  "plot_projection", "web_search", "explain_concept"],
                "memoria":       "Historial completo de la conversación en cada llamada.",
                "limitaciones":  ["No persiste estado entre sesiones", "No modifica el modelo estadístico"],
            },
        },
        "cointegracion": {
            "nombre": "Cointegración de Engle-Granger",
            "definicion": (
                "Dos series no estacionarias están cointegradas si existe una combinación lineal "
                "de ellas que sí es estacionaria. Se mueven juntas en el largo plazo."
            ),
            "test": "OLS de largo plazo → ADF sobre los residuales. Si residuales son estacionarios → cointegración.",
            "resultados_del_proyecto": {
                "Y ~ Equipo1": "p=0.013 ✓",
                "Z ~ Equipo1": "p=0.006 ✓",
                "Z ~ Equipo2": "p=0.008 ✓",
                "X ~ Equipo1": "p=0.500 ✗ → Price_X descartada",
            },
            "implicacion": (
                "La cointegración valida que modelar en niveles no produce regresión espuria "
                "y justifica el uso del ECM en lugar de modelar solo en diferencias."
            ),
        },
    }

    key = concepto.lower()
    for k, v in conceptos.items():
        if k in key or key in k:
            return json.dumps(v, ensure_ascii=False, indent=2)

    return f"Concepto '{concepto}' no encontrado. Disponibles: {list(conceptos.keys())}"


# ══════════════════════════════════════════════════════════════════════════════
# DEFINICIÓN DE TOOLS PARA LA API
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "get_projection",
        "description": (
            "Devuelve la proyección mensual de costos (media, IC 80% e IC 95%) "
            "para un equipo. Usar ante preguntas sobre pronósticos, valores esperados "
            "o costos futuros de un equipo."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "equipo": {"type": "string",
                           "description": "Nombre del equipo. Ej: 'Equipo1', 'equipo2'."}
            },
            "required": ["equipo"]
        }
    },
    {
        "name": "get_model_summary",
        "description": (
            "Retorna métricas del modelo (MAPE, R², MAE, RMSE), variables seleccionadas "
            "y metodología. Usar ante preguntas sobre calidad del modelo o cómo funciona."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "compare_scenarios",
        "description": (
            "Genera escenarios optimista, base y pesimista basados en los IC del bootstrap. "
            "Usar ante preguntas sobre rangos presupuestales o mejor/peor caso."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "equipo": {"type": "string", "description": "Nombre del equipo."}
            },
            "required": ["equipo"]
        }
    },
    {
        "name": "plot_projection",
        "description": (
            "Genera un gráfico PNG de la proyección con intervalos de confianza. "
            "Usar cuando pidan ver, graficar o visualizar los pronósticos."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "equipo": {"type": "string", "description": "Nombre del equipo a graficar."}
            },
            "required": ["equipo"]
        }
    },
    {
        "name": "web_search",
        "description": (
            "Busca noticias y contexto de mercado sobre materias primas o macroeconomía. "
            "Usar cuando pregunten por tendencias actuales, noticias o contexto externo "
            "que pueda afectar los precios."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Consulta en inglés. Ej: 'zinc price outlook 2024'."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "explain_concept",
        "description": (
            "Explica conceptos técnicos del análisis. Conceptos disponibles: "
            "'ecm' (Error Correction Model), 'agente' (IA convencional vs agente de IA), "
            "'cointegracion'. Usar ante preguntas metodológicas del evaluador."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "concepto": {"type": "string",
                             "description": "Concepto: 'ecm', 'agente' o 'cointegracion'."}
            },
            "required": ["concepto"]
        }
    },
]

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Eres un consultor cuantitativo especializado en gestión de costos para proyectos de construcción.
Tienes acceso a un modelo ECM (Error Correction Model) que proyecta costos de equipos a partir
de precios de materias primas.

CONTEXTO DEL MODELO:
- Equipo 1 → driver: Price_Y | MAPE: 1.72% | R² test: 0.987
- Equipo 2 → driver: Price_Z | MAPE: 3.24% | R² test: 0.922
- Horizonte: 12 meses (Sep 2023 – Ago 2024)
- Price_X descartada (sin cointegración)

REGLAS:
1. Usa herramientas para responder con datos reales, no de memoria.
2. Ante preguntas de mercado, usa web_search y combina con el pronóstico cuantitativo.
3. Sé conciso y orientado a decisiones. El usuario es gerente de proyecto.
4. Ante preguntas técnicas (ECM, agente, cointegración), usa explain_concept y elabora.
5. Si generas un gráfico, indica la ruta donde quedó guardado.
""".strip()

# ══════════════════════════════════════════════════════════════════════════════
# DESPACHADOR LOCAL
# ══════════════════════════════════════════════════════════════════════════════

def dispatch_tool(name: str, inputs: dict) -> str:
    dispatch = {
        "get_projection":   lambda: get_projection(**inputs),
        "get_model_summary": lambda: get_model_summary(),
        "compare_scenarios": lambda: compare_scenarios(**inputs),
        "plot_projection":  lambda: plot_projection(**inputs),
        "explain_concept":  lambda: explain_concept(**inputs),
    }
    fn = dispatch.get(name)
    return fn() if fn else f"Herramienta '{name}' no reconocida."

# ══════════════════════════════════════════════════════════════════════════════
# AGENTIC LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_agent():
    conversation = []
    print("\n" + "═" * 55)
    print("  Agente de Costos Operativos — Proyecto Construcción")
    print("═" * 55)
    print("Preguntas de ejemplo:")
    print("  · ¿Cuánto costará el Equipo 1 en marzo 2024?")
    print("  · Muéstrame los escenarios del Equipo 2")
    print("  · Grafica la proyección del Equipo 1")
    print("  · ¿Qué noticias hay sobre el zinc?")
    print("  · Explícame qué es un ECM")
    print("  · ¿Cuál es la diferencia entre IA convencional y un agente?")
    print("\nEscribe 'salir' para terminar.\n")

    while True:
        user_input = input("Tú: ").strip()
        if user_input.lower() in ("salir", "exit", "quit", "q"):
            print("Cerrando agente.")
            break
        if not user_input:
            continue

        conversation.append({"role": "user", "content": user_input})

        # Agentic loop — itera hasta stop_reason == "end_turn"
        while True:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=conversation,
            )

            conversation.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                tool_results = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    fn_name = block.name
                    fn_args = block.input
                    print(f"  [tool] {fn_name}({fn_args})")

                    if fn_name == "web_search":
                        # Búsqueda web via herramienta nativa de Anthropic
                        search_resp = client.messages.create(
                            model="claude-haiku-4-5-20251001",
                            max_tokens=1024,
                            tools=[WEB_SEARCH_TOOL],
                            messages=[{"role": "user", "content": fn_args["query"]}],
                        )
                        result = " ".join(
                            b.text for b in search_resp.content if hasattr(b, "text")
                        ) or "Sin resultados."
                    else:
                        result = dispatch_tool(fn_name, fn_args)

                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result,
                    })

                conversation.append({"role": "user", "content": tool_results})

            else:
                # end_turn → imprimir respuesta final
                final = "".join(
                    b.text for b in response.content if hasattr(b, "text")
                )
                print(f"\nAgente: {final}\n")
                print("─" * 55)
                break


if __name__ == "__main__":
    run_agent()