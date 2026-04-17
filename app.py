import streamlit as st
import os
import threading
from state import AgentState
from orchestrator import run_pipeline

st.set_page_config(page_title="AI Data Science Agent", page_icon="🤖", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0e1117; }
    .stApp { background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 100%); }
    .metric-card {
        background: linear-gradient(135deg, #1e2640, #252b40);
        border: 1px solid #3a4060;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .stage-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin: 4px;
    }
    .badge-done { background: #1a3d2e; color: #4ade80; border: 1px solid #4ade80; }
    .badge-running { background: #2e2a10; color: #facc15; border: 1px solid #facc15; }
    .badge-pending { background: #1e2640; color: #94a3b8; border: 1px solid #3a4060; }
    .badge-failed { background: #3d1a1a; color: #f87171; border: 1px solid #f87171; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("# 🤖 Agentic Data Science Pipeline")
st.markdown(
    "An autonomous AI workflow powered by LLMs. The **Coder** writes Python, "
    "the **Executor** runs it safely, and the **Reviewer** self-heals errors automatically."
)
st.divider()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")
    uploaded_file = st.file_uploader("1. Upload CSV Dataset", type=["csv"])

    if uploaded_file:
        st.success(f"✅ Loaded: `{uploaded_file.name}`")

    goal = st.text_area(
        "2. Define the Goal",
        placeholder="e.g. Perform EDA and predict customer churn using the best model.",
        height=100
    )
    target = st.text_input(
        "3. Target Column (leave blank to auto-detect)",
        placeholder="e.g. Attrition_Flag"
    )
    st.divider()
    start_btn = st.button("🚀 Start Agents", width='stretch', type="primary")
    st.markdown("---")
    st.markdown("**Powered by:**")
    st.markdown("🦙 Groq Llama 3.3-70b · 🤖 Multi-Agent Loop · 🔁 Self-Healing Reflexion")

# ── Stage progress helper ────────────────────────────────────────────────────
def render_stages(stage_states: dict):
    cols = st.columns(5)
    labels = ["Init", "EDA", "Cleaning", "Modeling", "Report"]
    icons = ["🏁", "📊", "🧹", "🧠", "📄"]
    for i, (label, icon) in enumerate(zip(labels, icons)):
        key = label.lower()
        s = stage_states.get(key, "pending")
        badge = {"done": "badge-done", "running": "badge-running", "failed": "badge-failed"}.get(s, "badge-pending")
        symbol = {"done": "✅", "running": "⏳", "failed": "❌", "pending": "⬜"}[s]
        cols[i].markdown(f'<div class="stage-badge {badge}">{icon} {label} {symbol}</div>', unsafe_allow_html=True)

# ── Main logic ───────────────────────────────────────────────────────────────
if start_btn:
    if not uploaded_file:
        st.warning("⚠️ Please upload a CSV file before starting.")
    elif not goal:
        st.warning("⚠️ Please specify a goal before starting.")
    else:
        # Clean up old artifacts
        for f in os.listdir('.'):
            if f.endswith('.png') and "architecture" not in f and "flowchart" not in f:
                try: os.remove(f)
                except: pass
        if os.path.exists("temp_dataset.csv"):
            os.remove("temp_dataset.csv")

        # Save uploaded CSV
        csv_path = "temp_dataset.csv"
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Initialize state
        state = AgentState(csv_path=csv_path, user_goal=goal, target_column=target)

        # Live log container
        st.markdown("### 📡 Live Agent Feed")
        log_container = st.container()
        log_lines = []

        stage_box = st.empty()
        stage_states = {"init": "done", "eda": "pending", "cleaning": "pending", "modeling": "pending", "report": "pending"}

        with stage_box:
            render_stages(stage_states)

        def push_update(msg: str):
            log_lines.append(msg)
            with log_container:
                for line in log_lines[-10:]:   # show last 10 lines
                    st.markdown(f"› {line}")
            # Update stage badges based on message content
            for stage in ["eda", "cleaning", "modeling"]:
                if stage in msg.lower():
                    if "✅" in msg:
                        stage_states[stage] = "done"
                    elif "❌" in msg or "failed" in msg.lower():
                        stage_states[stage] = "failed"
                    else:
                        stage_states[stage] = "running"
            if "report" in msg.lower():
                stage_states["report"] = "running"
            if "complete" in msg.lower():
                stage_states["report"] = "done"
            with stage_box:
                render_stages(stage_states)

        with st.spinner("Agents are working..."):
            final_state = run_pipeline(state, status_callback=push_update)

        # ── Results ──────────────────────────────────────────────────────────
        if final_state.error_message:
            st.error(f"❌ Pipeline Failed\n\n{final_state.error_message}")
        else:
            st.success("🎉 Pipeline completed successfully!")

            # Metric highlights
            if final_state.model_metrics:
                st.markdown("### 📈 Model Metrics")
                cols = st.columns(len(final_state.model_metrics))
                for i, (k, v) in enumerate(final_state.model_metrics.items()):
                    with cols[i]:
                        st.markdown(f'<div class="metric-card"><div style="color:#94a3b8;font-size:12px">{k}</div><div style="color:#60a5fa;font-size:28px;font-weight:700">{v}</div></div>', unsafe_allow_html=True)
                st.markdown("")

            # PDF Download Button
            if final_state.report_path and os.path.exists(final_state.report_path):
                with open(final_state.report_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download Full PDF Report",
                        data=pdf_file,
                        file_name="ds_agent_report.pdf",
                        mime="application/pdf",
                        width='stretch',
                    )

            # ── Code Tabs ────────────────────────────────────────────────────
            st.divider()
            tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🧹 Cleaning", "🧠 Modeling", "📈 Charts"])

            with tab1:
                st.subheader("Generated EDA Code")
                st.code(final_state.eda_code, language="python")
                if final_state.eda_output:
                    st.subheader("Execution Output")
                    st.text(final_state.eda_output[:2000])

            with tab2:
                st.subheader("Generated Cleaning Code")
                st.code(final_state.cleaning_code, language="python")
                if final_state.cleaning_output:
                    st.subheader("Execution Output")
                    st.text(final_state.cleaning_output[:2000])

            with tab3:
                st.subheader("Generated Modeling Code")
                st.code(final_state.model_code, language="python")
                if final_state.model_output:
                    st.subheader("Execution Output")
                    st.text(final_state.model_output[:2000])

            with tab4:
                png_files = [
                    f for f in sorted(os.listdir('.'))
                    if f.endswith('.png') and "architecture" not in f and "flowchart" not in f
                ]
                if png_files:
                    display_files = png_files[:12]
                    if len(png_files) > 12:
                        st.info(f"Showing 12 of {len(png_files)} generated plots.")
                    cols = st.columns(2)
                    for i, png in enumerate(display_files):
                        cols[i % 2].image(png, caption=png, use_container_width=True)
                else:
                    st.info("No plots were generated yet.")
