"""Execute the review notebook locally and embed the outputs into the .ipynb JSON."""
import io
import json
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from IPython.core.interactiveshell import InteractiveShell

def execute_notebook():
    nb_path = Path("kaggle_review_notebook/deepnt_plus_review.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    shell = InteractiveShell.instance()
    # Mock Kaggle environment if not present
    shell.run_cell("import os; os.environ['KAGGLE_KERNEL_RUN_TYPE'] = 'Interactive'")

    exec_count = 1
    for cell in nb_data["cells"]:
        if cell["cell_type"] != "code":
            continue

        code = "".join(cell["source"])
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            result = shell.run_cell(code)

        stdout_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()

        outputs = []
        if stdout_text:
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": stdout_text.splitlines(keepends=True)
            })
        if stderr_text:
            outputs.append({
                "output_type": "stream",
                "name": "stderr",
                "text": stderr_text.splitlines(keepends=True)
            })

        if result.error_in_exec:
            err = result.error_in_exec
            outputs.append({
                "output_type": "error",
                "ename": type(err).__name__,
                "evalue": str(err),
                "traceback": [str(err)]
            })
            print(f"❌ Error in cell {exec_count}: {err}")
        else:
            print(f"✅ Cell {exec_count} executed successfully.")

        cell["outputs"] = outputs
        cell["execution_count"] = exec_count
        exec_count += 1

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)

    print(f"🎉 Pre-executed outputs successfully embedded in {nb_path}!")

if __name__ == "__main__":
    execute_notebook()
