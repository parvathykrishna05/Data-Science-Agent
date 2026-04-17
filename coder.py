import os
import pandas as pd
from state import AgentState
from llm import call_llm


def auto_detect_target(csv_path: str, user_goal: str) -> str:
    """
    Uses the LLM to intelligently detect the most likely target column.
    """
    try:
        df = pd.read_csv(csv_path, nrows=3)
        columns = list(df.columns)
        sample = df.to_string(index=False, max_cols=10)
    except Exception as e:
        return ""

    system_prompt = (
        "You are a Data Science expert. Given a list of dataset columns and a user's goal, "
        "identify the single most likely TARGET column for a machine learning model. "
        "Reply with ONLY the exact column name. No explanation, no quotes."
    )
    user_message = f"User Goal: {user_goal}\nColumns: {columns}\nSample Data:\n{sample}"

    try:
        result = call_llm(system_prompt, user_message).strip().strip('"').strip("'")
        # Validate the result is an actual column name
        if result in columns:
            print(f"Auto-detected target column: '{result}'")
            return result
    except Exception:
        pass
    return ""


def generate_code(state: AgentState, stage: str) -> str:
    """
    Generates Data Science Python code based on the current pipeline stage.
    """
    # 1. Provide context about the dataset to the LLM
    try:
        df = pd.read_csv(state.csv_path, nrows=5)
        columns_info = df.dtypes.to_dict()
        sample_data = df.head(3).to_markdown()
    except Exception as e:
        columns_info = "Could not read CSV."
        sample_data = str(e)

    # 2. Construct the system prompt
    # Check if a cleaned dataset exists and has enough rows
    cleaned_path = "cleaned_dataset.csv"
    dataset_to_use = state.csv_path
    if stage == "modeling" and os.path.exists(cleaned_path):
        try:
            check_df = pd.read_csv(cleaned_path)
            if len(check_df) >= 50:
                dataset_to_use = cleaned_path
            else:
                print(f"WARNING: cleaned_dataset.csv only has {len(check_df)} rows. Falling back to original.")
        except Exception:
            pass

    system_prompt = (
        "You are an expert Data Scientist Python agent. Your task is to write clean, robust code.\n"
        "IMPORTANT RULES:\n"
        "1. Return ONLY valid, executable Python code. No markdown (no ```python). No explanations.\n"
        "2. Assume the dataset path is provided in the code.\n"
        "3. Import all necessary libraries (pandas, scikit-learn, matplotlib, seaborn, etc.).\n"
        "4. [CRITICAL] Save plots as PNG with descriptive names. Max 5 plots. Always call plt.close() after savefig().\n"
        "5. [CRITICAL] NEVER call plt.show(), plt.draw(), or input(). They cause timeouts.\n"
        "6. [CLEANING STAGE RULES]:\n"
        "   - NEVER use dropna() without subset=[target_column]. Only drop rows where the target is NaN.\n"
        "   - For all other columns: fillna with median() for numeric, mode()[0] for categorical.\n"
        "   - Print: print(f'Rows retained: {len(df)}') to confirm data is preserved.\n"
        "   - Save the result as cleaned_dataset.csv.\n"
        "7. [MODELING STAGE RULES]:\n"
        "   - Encode ALL categorical columns with LabelEncoder or get_dummies before training.\n"
        "   - Detect target type: if nunique() <= 20 use Classifier, else use Regressor.\n"
        "   - Use class_weight='balanced' for classifiers. Use StandardScaler on features.\n"
        "   - NEVER use GridSearchCV. Use simple .fit() only.\n"
        "   - If using cross_val_score, set cv=min(5, len(X_train)//10) to avoid split errors.\n"
        "   - Train 2 models minimum, print a comparison table, print accuracy/f1/r2."
    )

    # 3. Construct the user prompt
    user_prompt = f"""
    User Goal: {state.user_goal}
    Target Column: {state.target_column}
    Dataset Path: '{dataset_to_use}'

    Dataset Schema:
    {columns_info}

    Sample Data:
    {sample_data}

    Current Task: Write Python code for the '{stage}' stage.
    """

    # 4. If we are in a self-healing loop, provide the previous errors to fix
    if state.review_feedback and not state.review_passed:
        user_prompt += f"\n\n[CRITICAL] YOUR PREVIOUS CODE FAILED. PLEASE FIX IT BASED ON THIS FEEDBACK:\n{state.review_feedback}\n"

    # 5. Call the LLM
    response = call_llm(system_prompt, user_prompt)

    # Clean up formatting in case the LLM disobeys and uses markdown
    response = response.strip()
    if response.startswith("```python"):
        response = response[9:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]

    return response.strip()
