"""
streamlit_app.py — Chat UI for the operational cost agent.
"""

import io
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()  # loads ANTHROPIC_API_KEY from .env if present

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from anthropic import Anthropic

# ── path setup ──────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "app"))

CSV_PATH = os.path.join(_ROOT, "docs", "outputs", "proyeccion_equipos.csv")

# ── re-use tool implementations from agent.py ────────────────────────────────
from agent import (
    SYSTEM_PROMPT,
    TOOLS,
    WEB_SEARCH_TOOL,
    get_projection,
    get_model_summary,
    compare_scenarios,
    explain_concept,
)

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("⚠️ **ANTHROPIC_API_KEY** no encontrada. Crea un archivo `.env` en la raíz del proyecto con:\n```\nANTHROPIC_API_KEY=sk-ant-...\n```")
    st.stop()

client = Anthropic()

# ── plot helper — returns a BytesIO PNG instead of opening a file ────────────
def plot_projection_inline(equipo: str) -> tuple[str, bytes | None]:
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    except FileNotFoundError:
        return "ERROR: archivo de proyección no encontrado.", None

    mask = df["equipo"].str.lower().str.contains(equipo.lower())
    subset = df[mask]
    if subset.empty:
        return f"No hay datos para '{equipo}'.", None

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

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return f"Gráfico generado para {equipo}.", buf.read()


def dispatch_tool(name: str, inputs: dict) -> tuple[str, bytes | None]:
    """Returns (text_result, optional_image_bytes)."""
    if name == "get_projection":
        return get_projection(**inputs), None
    if name == "get_model_summary":
        return get_model_summary(), None
    if name == "compare_scenarios":
        return compare_scenarios(**inputs), None
    if name == "plot_projection":
        return plot_projection_inline(**inputs)
    if name == "explain_concept":
        return explain_concept(**inputs), None
    return f"Herramienta '{name}' no reconocida.", None


# ── agentic loop ─────────────────────────────────────────────────────────────
def run_turn(conversation: list, status_placeholder) -> tuple[str, list[bytes]]:
    """
    Runs the full agentic loop for one user turn.
    Returns (final_text, list_of_image_bytes).
    """
    images: list[bytes] = []

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
                status_placeholder.info(f"Ejecutando herramienta: **{fn_name}**…")

                if fn_name == "web_search":
                    search_resp = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=1024,
                        tools=[WEB_SEARCH_TOOL],
                        messages=[{"role": "user", "content": fn_args["query"]}],
                    )
                    result = " ".join(
                        b.text for b in search_resp.content if hasattr(b, "text")
                    ) or "Sin resultados."
                    img_bytes = None
                else:
                    result, img_bytes = dispatch_tool(fn_name, fn_args)

                if img_bytes:
                    images.append(img_bytes)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            conversation.append({"role": "user", "content": tool_results})

        else:
            final_text = "".join(
                b.text for b in response.content if hasattr(b, "text")
            )
            status_placeholder.empty()
            return final_text, images


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agente de Costos",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Agente de Costos Operativos")
st.caption("Hola! Soy el agente de costos operativos, te puedo ayudar con proyecciones ECM para Equipo 1 y Equipo 2, explicarte como funciona el modelo, las herramientas que puedo usar y cualquier otra pregunta que tengas.")

# session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "messages" not in st.session_state:
    st.session_state.messages = []   # {role, content, images?}

# sidebar
with st.sidebar:
    st.header("Preguntas de ejemplo")
    examples = [
        "¿Cuánto costará el Equipo 1 en marzo 2024?",
        "Muéstrame los escenarios del Equipo 2",
        "Grafica la proyección del Equipo 1",
        "¿Qué noticias hay sobre el zinc?",
        "Explícame qué es un ECM",
        "¿Cuál es la diferencia entre IA convencional y un agente?",
        "Resumen del modelo",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.prefill = ex

    st.divider()
    if st.button("🗑 Limpiar conversación", use_container_width=True):
        st.session_state.conversation = []
        st.session_state.messages = []
        st.rerun()

    st.markdown("**Modelos disponibles**")
    st.markdown("- Equipo 1 · MAPE 1.72%")
    st.markdown("- Equipo 2 · MAPE 3.24%")

# render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for img in msg.get("images", []):
            st.image(img)

# input — supports sidebar button prefill
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Escribe tu pregunta…", key="chat_input") or prefill

if user_input:
    # display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # add to raw conversation for the API
    st.session_state.conversation.append({"role": "user", "content": user_input})

    # run agent
    with st.chat_message("assistant"):
        status = st.empty()
        status.info("Procesando…")
        try:
            answer, images = run_turn(st.session_state.conversation, status)
        except Exception as e:
            answer = f"Error al ejecutar el agente: {e}"
            images = []

        st.markdown(answer)
        for img in images:
            st.image(img)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "images": images,
    })
