"""
report.py — Generates a professional PDF report from the AgentState.
Uses fpdf2 (already in requirements.txt).
"""

import os
from fpdf import FPDF
from state import AgentState


class AgentReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(30, 30, 50)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, "  AI Data Science Agent — Pipeline Report", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(240, 240, 250)
        self.set_text_color(30, 30, 80)
        self.cell(0, 8, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", size=9)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def code_block(self, code: str):
        self.set_font("Courier", size=7)
        self.set_fill_color(250, 248, 240)
        self.set_text_color(60, 60, 60)
        # Truncate very long code to avoid a massive PDF
        truncated = code[:3000] + "\n\n... [truncated]" if len(code) > 3000 else code
        self.multi_cell(0, 4, truncated, fill=True)
        self.ln(3)

    def metric_table(self, metrics: dict):
        if not metrics:
            return
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(200, 220, 255)
        self.set_text_color(20, 20, 80)
        self.cell(80, 7, "Metric", border=1, fill=True)
        self.cell(100, 7, "Value", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

        self.set_font("Helvetica", size=9)
        self.set_fill_color(245, 248, 255)
        self.set_text_color(40, 40, 40)
        for key, value in metrics.items():
            self.cell(80, 6, str(key), border=1, fill=True)
            self.cell(100, 6, str(value), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)


def generate_report(state: AgentState, output_path: str = "ds_agent_report.pdf") -> str:
    """
    Generates a complete PDF report from the AgentState and returns the output path.
    """
    pdf = AgentReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Cover Info ──────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 30, 80)
    pdf.ln(4)
    pdf.cell(0, 10, "Automated Data Science Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Goal: {state.user_goal}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Dataset: {os.path.basename(state.csv_path)}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Target Column: {state.target_column or 'Auto-detected'}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Review Cycles: {state.review_iterations}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # ── Model Metrics ──────────────────────────────────────────────────────────
    if state.model_metrics:
        pdf.section_title("Model Performance Metrics")
        pdf.metric_table(state.model_metrics)

    # ── Charts ─────────────────────────────────────────────────────────────────
    chart_files = [
        f for f in sorted(os.listdir('.'))
        if f.endswith('.png') and "architecture" not in f and "flowchart" not in f
    ]
    if chart_files:
        pdf.add_page()
        pdf.section_title("Generated Visualizations")
        for i, chart in enumerate(chart_files[:8]):   # max 8 charts in PDF
            try:
                if i % 2 == 0 and i > 0:
                    pdf.add_page()
                pdf.body_text(f"Chart: {chart}")
                pdf.image(chart, w=170)
                pdf.ln(4)
            except Exception:
                pdf.body_text(f"[Could not embed {chart}]")

    # ── Code Appendix ──────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Appendix A: EDA Code")
    pdf.code_block(state.eda_code or "(not generated)")
    pdf.body_text(f"Output:\n{state.eda_output[:500] if state.eda_output else '(none)'}")

    pdf.add_page()
    pdf.section_title("Appendix B: Data Cleaning Code")
    pdf.code_block(state.cleaning_code or "(not generated)")
    pdf.body_text(f"Output:\n{state.cleaning_output[:500] if state.cleaning_output else '(none)'}")

    pdf.add_page()
    pdf.section_title("Appendix C: Modeling Code")
    pdf.code_block(state.model_code or "(not generated)")
    pdf.body_text(f"Output:\n{state.model_output[:1000] if state.model_output else '(none)'}")

    # ── Save ───────────────────────────────────────────────────────────────────
    pdf.output(output_path)
    state.report_path = output_path
    print(f"Report saved to: {output_path}")
    return output_path
