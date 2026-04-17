import subprocess
import tempfile
import os
from typing import Tuple

def execute_code(code: str, timeout: int = 120) -> Tuple[bool, str, str]:
    """
    Safely executes Python code in a separate subprocess.
    Returns: (success: bool, stdout: str, stderr: str)
    """
    # Create a temporary file to hold the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code)
        temp_file = f.name
        
    try:
        # Run the temporary Python file in a new process
        result = subprocess.run(
            ["python", temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Execution timed out after {} seconds.".format(timeout)
    except Exception as e:
        return False, "", f"Execution failed: {str(e)}"
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
