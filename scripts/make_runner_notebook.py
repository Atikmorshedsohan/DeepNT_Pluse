"""Build DeepNT_Model_Benchmark_Runner.ipynb containing run_kaggle.py and its benchmark execution outputs."""
import json
from pathlib import Path
import nbformat as nbf
import pandas as pd

def build_runner_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.10.0"
    }

    c1 = nbf.v4.new_markdown_cell("""# 🚀 DeepNT+ 5-Seed Benchmark Runner & Training Suite

**Dataset**: [DeepNT-Plus Network Tomography Suite](https://www.kaggle.com/datasets/atikmorshedsohan/deepnt-plus-network-tomography-suite)  
**Execution Environment**: Kaggle Tesla T4 GPU (CUDA 13.0, PyTorch 2.11+)  
**Evaluation Protocol**: 5 Random Seeds (`[0, 1, 2, 3, 4]`), 35 Epochs per run, Multi-Episode Training.

This notebook contains the complete GPU execution script used to produce the master benchmark results comparing:
1. **IPv4 10-Graphs (GeoLite2 Side-Information Track)**: `edge_dim=2` (Haversine distance).
2. **IPv4 10-Graphs (Pure Topological Track)**: `edge_dim=1` (log path counts only, zero geo leakage).
3. **IPv6 10-Graphs (GeoLite2 Side-Information Track)**: `edge_dim=2`.
4. **IPv6 10-Graphs (Pure Topological Track)**: `edge_dim=1`.
5. **IPv4 Legacy Suites**: CAIDA run_03 (Dense) & CAIDA run_02 (15-Graphs).""")

    with open("kaggle_kernel_run/run_kaggle.py", "r", encoding="utf-8") as f:
        code_text = f.read()

    # Read log text
    log_text = ""
    log_file = Path("benchmark_results/deepnt-model-benchmark-suite.log")
    if log_file.exists():
        try:
            raw = log_file.read_text(encoding="utf-8")
            events = json.loads(raw) if raw.strip().startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
            for ev in events:
                if ev.get("stream_name") in ("stdout", "stderr"):
                    log_text += ev.get("data", "")
        except Exception:
            pass

    c2 = nbf.v4.new_code_cell(code_text)
    c2["execution_count"] = 1
    if log_text:
        c2.outputs = [nbf.v4.new_output(
            output_type="stream",
            name="stdout",
            text=log_text.splitlines(keepends=True)
        )]

    c3 = nbf.v4.new_markdown_cell("""## 📊 Final Master Benchmark Summary Table""")

    summary_code = """import pandas as pd
from pathlib import Path
from IPython.display import display

df = pd.read_csv('benchmark_results/authentic_master_benchmark_summary.csv')
display(df[['Dataset', 'Model', 'Family', 'MAE', 'MAE_std', 'R2', 'Cov90', 'ECE']].round(3))"""

    c4 = nbf.v4.new_code_cell(summary_code)
    c4["execution_count"] = 2
    df_sum = pd.read_csv("benchmark_results/authentic_master_benchmark_summary.csv")
    c4.outputs = [nbf.v4.new_output(
        output_type="execute_result",
        execution_count=2,
        data={
            "text/plain": df_sum[['Dataset', 'Model', 'Family', 'MAE', 'MAE_std', 'R2', 'Cov90', 'ECE']].round(3).to_string(index=False),
            "text/html": df_sum[['Dataset', 'Model', 'Family', 'MAE', 'MAE_std', 'R2', 'Cov90', 'ECE']].round(3).to_html(index=False)
        }
    )]

    nb.cells = [c1, c2, c3, c4]
    out_path = Path("DeepNT_Model_Benchmark_Runner.ipynb")
    nbf.write(nb, str(out_path))
    print(f"✅ Successfully created {out_path}")

if __name__ == "__main__":
    build_runner_notebook()
