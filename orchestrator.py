import logging
from state import AgentState
from coder import generate_code, auto_detect_target
from executor import execute_code
from reviewer import review_code
from report import generate_report

# ── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agent_run.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

MAX_RETRIES = 5


def run_stage(state: AgentState, stage: str, status_callback=None) -> bool:
    """
    Runs a single stage through the self-healing loop: Coder -> Executor -> Reviewer.
    Returns True if successful, False if it failed after max retries.
    An optional status_callback(msg: str) can be used to push live updates to the UI.
    """
    state.current_stage = stage
    state.review_passed = False
    state.review_iterations = 0
    state.review_feedback = ""

    log.info(f"--- Starting Stage: {stage.upper()} ---")
    if status_callback:
        status_callback(f"🔄 Starting **{stage.upper()}** stage...")

    while state.review_iterations < MAX_RETRIES and not state.review_passed:
        attempt_num = state.review_iterations + 1
        log.info(f"Iteration {attempt_num}/{MAX_RETRIES} for {stage}...")
        if status_callback:
            status_callback(f"✏️ Coder is writing `{stage}` code (attempt {attempt_num}/{MAX_RETRIES})...")

        # 1. Coder generates code
        code = generate_code(state, stage)

        if stage == "eda":
            state.eda_code = code
        elif stage == "cleaning":
            state.cleaning_code = code
        elif stage == "modeling":
            state.model_code = code

        if status_callback:
            status_callback(f"⚙️ Executor is running `{stage}` code...")

        # 2. Executor runs code
        success, stdout, stderr = execute_code(code)

        if stage == "eda":
            state.eda_output = stdout
        elif stage == "cleaning":
            state.cleaning_output = stdout
        elif stage == "modeling":
            state.model_output = stdout

        if status_callback:
            status_callback(f"🔍 Reviewer is checking `{stage}` output...")

        # 3. Reviewer checks results
        review_code(state, stage, code, stdout, stderr)
        state.review_iterations += 1

        if state.review_passed:
            log.info(f"✓ {stage} passed review!")
            if status_callback:
                status_callback(f"✅ **{stage.upper()}** passed!")
        else:
            log.warning(f"✗ Review failed: {state.review_feedback[:120]}...")
            if status_callback:
                status_callback(f"⚠️ Review failed for `{stage}`. Self-healing...")

    if not state.review_passed:
        state.error_message = (
            f"Failed at stage '{stage}' after {MAX_RETRIES} attempts.\n"
            f"Last feedback: {state.review_feedback}"
        )
        log.error(state.error_message)
        return False

    return True


def run_pipeline(state: AgentState, status_callback=None) -> AgentState:
    """
    Runs the full end-to-end data science pipeline.
    """
    log.info(f"Pipeline started | Goal: {state.user_goal}")

    # Auto-detect target column if not provided
    if not state.target_column:
        if status_callback:
            status_callback("🧠 Auto-detecting target column...")
        state.target_column = auto_detect_target(state.csv_path, state.user_goal)

    stages = ["eda", "cleaning", "modeling"]
    for stage in stages:
        success = run_stage(state, stage, status_callback)
        if not success:
            return state

    # Generate the PDF report
    if status_callback:
        status_callback("📄 Generating PDF report...")
    try:
        generate_report(state)
        log.info(f"Report saved to: {state.report_path}")
    except Exception as e:
        log.warning(f"Report generation failed: {e}")

    log.info("Pipeline completed successfully!")
    if status_callback:
        status_callback("🎉 Pipeline complete!")

    return state
