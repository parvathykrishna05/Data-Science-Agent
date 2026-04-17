# 🤖 Agentic Data Science Pipeline (ds-agent)

An autonomous Data Science workflow powered by Large Language Models (LLMs). This project features a multi-agent system where agents collaborate to perform Exploratory Data Analysis (EDA), Data Cleaning, and Model Training autonomously.

## 🌟 Key Features

*   **Self-Healing Agent Loop**: Employs a robust Coder -> Executor -> Reviewer pipeline. If the generated code fails, the Reviewer analyzes the stack trace and feeds the error back to the Coder for auto-correction (Reflexion pattern).
*   **Safe Code Execution**: Executes LLM-generated Python code safely in isolated subprocesses.
*   **Interactive UI**: Built with Streamlit for a clean, user-friendly interface.
*   **State Management**: Uses a custom dataclass-based state machine to pass context seamlessly between agents.

## 🏗️ Architecture

The pipeline consists of the following components working sequentially:
1.  **Init**: Setup and dataset ingestion.
2.  **EDA**: Exploratory Data Analysis (Generates plots and basic stats).
3.  **Cleaning**: Handles missing values, outliers, and data formatting.
4.  **Modeling**: Trains a machine learning model and evaluates metrics.
5.  **Report**: (Phase 6 - WIP) Compiles findings into a final PDF.

*(Check the included architecture `.png` files in this repository for visual diagrams!)*

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
You must provide an Anthropic API Key. Open the `.env` file in the root directory and add:
```env
ANTHROPIC_API_KEY=sk-ant-your_api_key_here
```
*(Optional)* You can also override the default model by adding:
`ANTHROPIC_MODEL=claude-3-5-sonnet-20240620`

### 4. Running the Application
Launch the Streamlit interface:
```bash
streamlit run app.py
```

Upload your dataset (`.csv`), type in your specific goal (e.g., *"Predict customer churn using a Random Forest classifier"*), and watch the agents write, execute, and review the code for you!
