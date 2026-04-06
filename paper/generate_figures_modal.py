"""Generate paper figures on Modal GPU.

Runs the figure generation script on an A100 and downloads the PDFs.

Usage:
    modal run paper/generate_figures_modal.py
"""

import modal

app = modal.App("jax-lbm-figures")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]",
        "jaxlib",
        "optax",
        "matplotlib",
    )
    .add_local_dir("jax_lbm", "/root/jax_lbm", copy=True)
    .add_local_dir("paper", "/root/paper", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .run_commands("cd /root && pip install -e .")
)


@app.function(image=image, gpu="A100", timeout=1800)
def generate_figures():
    """Run generate_figures.py on GPU and return the PDFs."""
    import subprocess
    import os

    os.makedirs("/root/paper/figures", exist_ok=True)

    result = subprocess.run(
        ["python", "paper/generate_figures.py"],
        capture_output=True,
        text=True,
        cwd="/root",
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Read generated figures and return them.
    figures = {}
    fig_dir = "/root/paper/figures"
    for name in os.listdir(fig_dir):
        path = os.path.join(fig_dir, name)
        with open(path, "rb") as f:
            figures[name] = f.read()
        print(f"Generated: {name} ({len(figures[name])} bytes)")

    return figures


@app.local_entrypoint()
def main():
    import os

    figures = generate_figures.remote()

    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)

    for name, data in figures.items():
        path = os.path.join(out_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        print(f"Saved {path} ({len(data)} bytes)")
