import re
from state import AgentState
from llm import call_llm

# Patterns that are harmless warnings, NOT actual errors
IGNORABLE_PATTERNS = [
    r"UserWarning",
    r"FutureWarning",
    r"DeprecationWarning",
    r"RuntimeWarning",
    r"ConvergenceWarning",
    r"tight_layout",
    r"Matplotlib",
    r"MatplotlibDeprecationWarning",
    r"DataConversionWarning",
]


def is_real_error(stderr: str) -> bool:
    """
    Returns True only if stderr contains an actual crash (Traceback),
    not just harmless warnings.
    """
    if not stderr or not stderr.strip():
        return False

    # Check if there's a real Traceback with an actual Exception
    has_traceback = "Traceback (most recent call last)" in stderr
    has_exception = bool(re.search(
        r"(Error|Exception|ImportError|ModuleNotFoundError|ValueError|KeyError|TypeError|IndexError|FileNotFoundError|ZeroDivisionError):",
        stderr
    ))

    if has_traceback and has_exception:
        return True

    # If it's just warnings, it's not a real error
    lines = [l.strip() for l in stderr.strip().split("\n") if l.strip()]
    for line in lines:
        is_ignorable = any(re.search(pat, line, re.IGNORECASE) for pat in IGNORABLE_PATTERNS)
        if not is_ignorable and not line.startswith("  "):
            # Found a non-warning, non-indentation line that isn't ignorable
            if "Error" in line or "Exception" in line:
                return True

    return False


def review_code(state: AgentState, stage: str, code: str, stdout: str, stderr: str) -> None:
    """
    Reviews the execution results of the generated code.
    Updates the state with whether it passed and feedback for the coder.
    """
    # 1. Only fail on REAL errors, not warnings
    if is_real_error(stderr):
        state.review_passed = False
        state.review_feedback = f"Execution failed with Python error:\n{stderr[-1000:]}\n\nPlease fix the code."
        return

    # 2. If there's no stdout at all, that's suspicious
    if not stdout or not stdout.strip():
        state.review_passed = False
        state.review_feedback = "The code produced no output at all. Please add print statements for results and metrics."
        return

    # 3. Use LLM to review the logical output
    system_prompt = (
        "You are an expert Data Science Code Reviewer.\n"
        "You are reviewing code from a MULTI-STAGE pipeline: EDA → Cleaning → Modeling.\n"
        "The user's high-level goal covers ALL stages. Each stage has its own purpose:\n"
        "  - 'eda' stage: Should produce statistics, visualizations, and insights about the data.\n"
        "  - 'cleaning' stage: Should handle missing values, encode categoricals, and save a clean CSV.\n"
        "  - 'modeling' stage: Should train ML models and print metrics. THIS IS EXPECTED even if the user only said 'perform EDA'.\n\n"
        "RULES:\n"
        "1. If the code for THIS SPECIFIC STAGE ran correctly and produced reasonable output, reply EXACTLY with 'PASS'.\n"
        "2. If there are real problems (0.0 accuracy, NaN metrics, empty output, crashes), reply with 'FAIL: <reason>'.\n"
        "3. NEVER reject a stage because it doesn't match the user's high-level goal. Each stage does its own job.\n"
        "4. Ignore warnings about matplotlib, tight_layout, convergence, or deprecation.\n"
        "Do not write any other text."
    )

    user_prompt = f"""
    Pipeline Stage: {stage}
    User Goal: {state.user_goal}

    Generated Code:
    {code[:2000]}

    Standard Output from Execution:
    {stdout[:2000]}

    Warnings (ignorable):
    {stderr[:500] if stderr else '(none)'}

    Does this output indicate success?
    """

    response = call_llm(system_prompt, user_prompt).strip()

    # 3. Update state
    if response.upper().startswith("PASS"):
        state.review_passed = True
        state.review_feedback = ""
    else:
        state.review_passed = False
        state.review_feedback = response
