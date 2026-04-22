# Agente de costos operativos

Aplicación de **consultoría cuantitativa** para gestión de costos en proyectos de construcción. Combina pronósticos de un **modelo ECM** (Error Correction Model) con un **agente conversacional** basado en la API de Anthropic (Claude Haiku en este caso): el modelo decide cuándo consultar datos, comparar escenarios, graficar o buscar contexto de mercado en la web.

## Qué incluye

- **Interfaz web (Streamlit)** — chat con historial, gráficos embebidos y preguntas de ejemplo.
- **Agente por terminal** — mismo razonamiento y herramientas, ejecutable con `python app/agent.py`.
- **Docker** — imagen lista para servir la app Streamlit en el puerto **8501** o en CloudRun por ejemplo.

### Herramientas del agente

| Herramienta | Descripción |
|-------------|-------------|
| `get_projection` | Proyección mensual (media, IC 80% e IC 95%) por equipo desde CSV. |
| `get_model_summary` | Métricas (MAPE, R², MAE, RMSE), drivers y metodología. |
| `compare_scenarios` | Escenarios optimista / base / pesimista a partir de los intervalos del modelo. |
| `plot_projection` | Gráfico de proyección con intervalos (en Streamlit: imagen en el chat; en CLI: PNG bajo `docs/outputs/`). |
| `web_search` | Búsqueda web vía herramienta nativa de Anthropic. |
| `explain_concept` | Explicaciones estructuradas de ECM, agente de IA y cointegración. |

### Contexto del modelo (resumen)

- **Equipo 1** — driver: `Price_Y` · MAPE test ~1.72% · R² test ~0.987  
- **Equipo 2** — driver: `Price_Z` · MAPE test ~3.24% · R² test ~0.922  
- Horizonte de pronóstico en los datos: **12 meses** (p. ej. Sep 2023 – Ago 2024 en el dataset actual).  
- En el código aparecen nombres de serie tipo `Price_Equipo1` / `Price_Equipo2` en el CSV de proyección.

El modelo remoto usado en las llamadas a la API es **`claude-haiku-4-5-20251001`**, generando excelentes resultados a un costo muy bajo.

## Estructura del repositorio

```
costs_ops_agent/
├── app/
│   ├── agent.py           # Herramientas, system prompt, loop CLI
│   └── streamlit_app.py   # UI Streamlit y loop del agente
├── data/                  # Series históricas (X, Y, Z, histórico equipos)
├── docs/
│   ├── exploration.ipynb  # Análisis / generación de salidas (incluye proyecciones)
│   ├── outputs/
│   │   ├── proyeccion_equipos.csv   # Entrada principal del agente para pronósticos
│   │   └── outputs.png   # Imágenes para complementar las respuestas del agente
│   ├── **Informe_gestion_Costos_operativos.pdf**     # **INFORME SOLICITADO para el caso de estudio** documentación del modelado y proceso.
│   ├── **digrama_GCP.png**                           # **DIAGRAMA SOLICITADO para el caso de estudio**.
│   └── …                                             # PDFs u otros materiales del caso
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .env                   # OJO No versionar
```

## Requisitos

- **Python 3.11+** (la imagen Docker usa 3.11; entornos locales cercanos funcionan bien).
- Cuenta y **clave de API de Anthropic** con acceso a los modelos y herramientas, agregar el ANTHROPIC_API_KEY.

## Configuración

### 1. Clonar e instalar dependencias

```bash
cd costs_ops_agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Variable de entorno

Crea un archivo **`.env`** en la raíz del proyecto (está en `.gitignore`):

```env
ANTHROPIC_API_KEY=...
```

También se puede exportar la variable en la shell:

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

### 3. Datos de proyección

El agente lee **`docs/outputs/proyeccion_equipos.csv`**. Si falta, ejecuta o actualiza el flujo en **`docs/exploration.ipynb`** (o el proceso que genera ese CSV) antes de usar las herramientas de pronóstico.

## Uso

### Streamlit (recomendado para demo)

Desde la raíz del repositorio:

```bash
streamlit run app/streamlit_app.py
```

Abre la URL que muestre la terminal (por defecto `http://localhost:8501`).

### Agente en terminal

```bash
export ANTHROPIC_API_KEY="..."
python app/agent.py
```

Comandos para salir: `salir`, `exit`, `quit`, `q`.

**Nota:** La herramienta `plot_projection` en modo CLI intenta abrir el PNG con el visor del sistema (`open` en macOS). En Linux o Windows puede que necesites abrir manualmente el archivo bajo `docs/outputs/`.

## Docker

Construcción y ejecución:

```bash
docker build -t costs-ops-agent .
docker run --rm -p 8501:8501 -e ANTHROPIC_API_KEY="..." costs-ops-agent
```

La app queda en **http://localhost:8501**.

## Licencia y datos

Este repositorio es un proyecto de consultoría / caso de estudio. Revisar a profundidad este README.md y los PDFs asociados.