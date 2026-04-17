"""
core/state.py
Shared state dataclass passed between all agents throughout the pipeline.
Think of this as the "memory" the agents read from and write to.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class AgentState:
    # ── Inputs ──────────────────────────────────────────────────────────────
    csv_path: str = ""                  # Path to the uploaded CSV file
    user_goal: str = ""                 # e.g. "Predict customer churn"
    target_column: str = ""            # Detected or user-specified target

    # ── Coder Agent outputs ──────────────────────────────────────────────────
    eda_code: str = ""                  # Python code for EDA
    cleaning_code: str = ""            # Python code for data cleaning
    model_code: str = ""               # Python code for model training

    # ── Reviewer Agent outputs ───────────────────────────────────────────────
    review_passed: bool = False        # Did the reviewer approve the code?
    review_feedback: str = ""          # Reviewer's critique (if any)
    review_iterations: int = 0        # How many review cycles happened

    # ── Executor Agent outputs ───────────────────────────────────────────────
    eda_output: str = ""               # stdout/results from running EDA code
    cleaning_output: str = ""         # stdout/results from cleaning code
    model_output: str = ""            # stdout/results from model training
    model_metrics: dict = field(default_factory=dict)  # e.g. {"accuracy": 0.87}
    chart_paths: list = field(default_factory=list)    # Saved plot file paths
    execution_errors: list = field(default_factory=list)  # Any errors encountered

    # ── Report ────────────────────────────────────────────────────────────────
    report_path: str = ""              # Final PDF output path

    # ── Pipeline control ─────────────────────────────────────────────────────
    current_stage: str = "init"        # init → eda → cleaning → modeling → report
    error_message: str = ""            # Top-level error if pipeline fails

    def summary(self) -> str:
        """Human-readable pipeline status for logging."""
        lines = [
            f"Stage     : {self.current_stage}",
            f"Goal      : {self.user_goal}",
            f"CSV       : {self.csv_path}",
            f"Target    : {self.target_column or '(not yet detected)'}",
            f"Review    : {'✓ passed' if self.review_passed else '✗ pending'} (iterations: {self.review_iterations})",
            f"Metrics   : {self.model_metrics or '(none yet)'}",
            f"Charts    : {len(self.chart_paths)} saved",
            f"Report    : {self.report_path or '(not generated)'}",
        ]
        return "\n".join(lines)
