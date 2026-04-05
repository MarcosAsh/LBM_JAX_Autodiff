"""
Modal GPU worker for the differentiable LBM solver.

Runs examples and tests on an A100 GPU where JAX can actually fly.

Usage:
    modal run modal_worker.py --example channel      # Poiseuille flow
    modal run modal_worker.py --example sphere        # Sphere Cd at Re=100
    modal run modal_worker.py --example optimize      # Shape optimization
    modal run modal_worker.py --example inlet         # Inlet velocity optimization
    modal run modal_worker.py --tests                 # Run full test suite
"""

import modal

app = modal.App("jax-lbm")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]",
        "jaxlib",
        "optax",
        "pytest",
    )
    .add_local_dir("jax_lbm", "/root/jax_lbm", copy=True)
    .add_local_dir("tests", "/root/tests", copy=True)
    .add_local_dir("examples", "/root/examples", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .run_commands("cd /root && pip install -e .")
)


@app.function(image=image, gpu="A100", timeout=3600)
def run_example(name: str):
    """Run one of the example scripts on GPU."""
    import subprocess
    import sys

    examples = {
        "channel": "examples/01_channel_flow.py",
        "sphere": "examples/02_sphere_drag.py",
        "inlet": "examples/03_optimize_inlet.py",
        "optimize": "examples/04_shape_optimization.py",
        "sphere-mrt": "examples/05_sphere_cd_mrt.py",
        "optimize-adam": "examples/06_shape_optimization_adam.py",
        "bouzidi-vs-standard": "examples/07_bouzidi_vs_standard.py",
        "grad-convergence": "examples/08_gradient_convergence.py",
        "sphere-fine": "examples/09_sphere_cd_fine_grid.py",
    }

    if name not in examples:
        print(f"Unknown example: {name}")
        print(f"Available: {', '.join(examples.keys())}")
        sys.exit(1)

    script = examples[name]
    print(f"Running {script} on GPU...")
    print()

    result = subprocess.run(
        ["python", script],
        capture_output=False,
        cwd="/root",
    )
    return result.returncode


@app.function(image=image, gpu="A100", timeout=1800)
def run_tests(include_slow: bool = True):
    """Run the full test suite on GPU."""
    import subprocess

    cmd = ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
    if include_slow:
        cmd.extend(["-m", "slow or not slow"])

    print("Running test suite on GPU...")
    print()

    result = subprocess.run(cmd, capture_output=False, cwd="/root")
    return result.returncode


@app.local_entrypoint()
def main(
    example: str = "",
    tests: bool = False,
    slow: bool = True,
):
    if tests:
        rc = run_tests.remote(include_slow=slow)
        raise SystemExit(rc)
    elif example:
        rc = run_example.remote(example)
        raise SystemExit(rc)
    else:
        print("Usage:")
        print("  modal run modal_worker.py --example channel")
        print("  modal run modal_worker.py --example sphere")
        print("  modal run modal_worker.py --example optimize")
        print("  modal run modal_worker.py --example inlet")
        print("  modal run modal_worker.py --tests")
