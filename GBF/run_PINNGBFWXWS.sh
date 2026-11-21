#!/bin/bash
set -euo pipefail

# Wrapper for:
#   1. NumericalGBF.py (optional; writes ./Numerical/l{mode}/Output/numericalGBF.csv)
#   2. the PINN script (PINNGBFWX.py by default)
#
# All options are passed as named flags (--mass 0.5 --mode 2 ...) and forwarded unchanged to the PINN.
# The shared ones (mass, mode, x_extract, omega_start, omega_final, query_steps) are also given to NumericalGBF.py.
# Wrapper-only option:
#   --run_numerical yes|no   yes (default): run NumericalGBF.py first and compare the PINN against its CSV
#                            no: skip it and pass --no_numerical_csv to the PINN
#
# Environment overrides: PYTHON_BIN, PINN_SCRIPT, NUMERICAL_SCRIPT, REQUIRE_GPU (1 = refuse to run on CPU),
# GBF_BASIS_VERBOSE (1 = gbf_reference.py prints every numerical inward u1 basis integration).

# Run from the project directory containing this wrapper.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# CUDA 12.8 PyTorch build: has kernels for the RTX 2080 Ti (sm_75), 3090 (sm_86) and 5090 (sm_120) nodes.
PYTHON_BIN="${PYTHON_BIN:-/scratch/wiay/danielk/venvs/pinns-gpu-cu128/bin/python}"
PINN_SCRIPT="${PINN_SCRIPT:-PINNGBFWX.py}"
NUMERICAL_SCRIPT="${NUMERICAL_SCRIPT:-NumericalGBF.py}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"

echo "Python: $PYTHON_BIN"
echo "GBF_BASIS_VERBOSE: ${GBF_BASIS_VERBOSE:-0}"

REQUIRE_GPU="$REQUIRE_GPU" "$PYTHON_BIN" - <<'PY'
import os
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if not torch.cuda.is_available():
    if os.environ.get("REQUIRE_GPU", "1") == "1":
        raise SystemExit("ERROR: CUDA is unavailable; refusing to run the PINN on CPU.")
    print("WARNING: CUDA is unavailable; running on CPU (REQUIRE_GPU=0).")
else:
    print("GPU:", torch.cuda.get_device_name(0))
PY

# The PINN imports gbf_reference.py (including the basis-tracking functions) from its own directory
for required in "$PINN_SCRIPT" gbf_reference.py; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: $required not found in $SCRIPT_DIR" >&2
        exit 2
    fi
done
if ! grep -q "def basis_tracking_summary" gbf_reference.py; then
    echo "ERROR: gbf_reference.py has no basis-tracking functions; upload the version that matches $PINN_SCRIPT" >&2
    exit 2
fi

RUN_NUMERICAL="yes"
MASS=""
MODE=""
MODES=""              # wrapper-only: e.g. "2,3,4" runs every l in one job, lowest first, warm-starting each from the previous
WARM_ADAM_STEPS=""    # wrapper-only: Adam steps for the warm-started l (default: same as --adam_steps)
RUN_TAG=""
X_EXTRACT=""
OMEGA_START="1.2"      # PINN defaults, used if the flag is not given
OMEGA_FINAL="0.01"
QUERY_STEPS="200"
PINN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_numerical)   RUN_NUMERICAL="$2"; shift 2; continue ;;
        --modes)           MODES="$2"; shift 2; continue ;;
        --warm_adam_steps) WARM_ADAM_STEPS="$2"; shift 2; continue ;;
        --mass)            MASS="$2" ;;
        --mode)            MODE="$2" ;;
        --x_extract)       X_EXTRACT="$2" ;;
        --omega_start)     OMEGA_START="$2" ;;
        --omega_final)     OMEGA_FINAL="$2" ;;
        --query_steps)     QUERY_STEPS="$2" ;;
        --run_tag)         RUN_TAG="$2" ;;
    esac
    PINN_ARGS+=("$1")
    shift
done

if [[ -z "$MASS" || -z "$X_EXTRACT" || ( -z "$MODE" && -z "$MODES" ) || ( -n "$MODE" && -n "$MODES" ) ]]; then
    echo "Usage: $0 --mass M (--mode L | --modes \"L1 L2 ...\") --x_extract X [any other PINNGBFWX.py options] [--run_numerical yes|no] [--warm_adam_steps N]" >&2
    exit 2
fi

if [[ "$RUN_NUMERICAL" != "yes" && "$RUN_NUMERICAL" != "no" ]]; then
    echo "ERROR: --run_numerical must be yes or no (got '$RUN_NUMERICAL')" >&2
    exit 2
fi
[[ "$RUN_NUMERICAL" == "no" ]] && PINN_ARGS+=(--no_numerical_csv)

run_numerical_for() {
    # $1 = l. Writes ./Numerical/l$1/Output/numericalGBF.csv for the PINN comparison
    [[ "$RUN_NUMERICAL" == "yes" ]] || return 0
    echo "Starting $NUMERICAL_SCRIPT for l = $1 (query_steps = $QUERY_STEPS)"
    "$PYTHON_BIN" "$NUMERICAL_SCRIPT" --mass "$MASS" --mode "$1" --x_extract "$X_EXTRACT" \
        --omega_start "$OMEGA_START" --omega_final "$OMEGA_FINAL" --num_steps "$QUERY_STEPS"
}

if [[ -n "$MODE" ]]; then
    # Single l, as before: output in ./GBFWXData/l{mode}/{run_name}
    run_numerical_for "$MODE"
    echo "============================================================"
    echo "Starting $PINN_SCRIPT ${PINN_ARGS[*]}"
    echo "============================================================"
    "$PYTHON_BIN" "$PINN_SCRIPT" "${PINN_ARGS[@]}"
else
    # Multi l: ./GBFWXWSData/{timestamp}_M{mass}_x{x}[_{tag}]/l{L}/..., lowest l first, each warm-started from the previous
    MULTI_DIR="./GBFWXWSData/$(date +%Y%m%d_%H%M%S)_M${MASS}_x${X_EXTRACT}${RUN_TAG:+_${RUN_TAG}}"
    mkdir -p "$MULTI_DIR"
    echo "Multi-l run in $MULTI_DIR (l = $MODES)"
    PREV_CKPT=""
    for L in $(tr ' ,' '\n\n' <<< "$MODES" | grep -v '^$' | sort -n -u); do
        run_numerical_for "$L"
        L_ARGS=("${PINN_ARGS[@]}" --mode "$L" --output_dir "$MULTI_DIR/l$L")
        if [[ -n "$PREV_CKPT" ]]; then
            L_ARGS+=(--init_checkpoint "$PREV_CKPT")
            # argparse keeps the last occurrence, so this overrides --adam_steps for the warm-started l only
            [[ -n "$WARM_ADAM_STEPS" ]] && L_ARGS+=(--adam_steps "$WARM_ADAM_STEPS")
        fi
        echo "============================================================"
        echo "Starting l = $L: $PINN_SCRIPT ${L_ARGS[*]}"
        echo "============================================================"
        "$PYTHON_BIN" "$PINN_SCRIPT" "${L_ARGS[@]}"
        PREV_CKPT="$MULTI_DIR/l$L/network/trained_model.pth"
    done

    # Combined GBF figure for every l in the run
    "$PYTHON_BIN" - "$MULTI_DIR" "$MASS" <<'PY'
import csv, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix"})

multi_dir, mass = sys.argv[1], float(sys.argv[2])
runs = sorted((p for p in glob.glob(os.path.join(multi_dir, "l*")) if os.path.isdir(p)), key = lambda p: int(os.path.basename(p)[1:]))

def read(path, xkey, ykey):
    with open(path, newline = "") as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r[xkey]) for r in rows]), np.array([float(r[ykey]) for r in rows])

fig, axes = plt.subplots(1, 2, figsize = (14, 5.5))
summary = []
for path in runs:
    l = int(os.path.basename(path)[1:])
    omega, gbf = read(os.path.join(path, "results", "PINN_query_frequencies.csv"), "omega", "PINN_GBF")
    Mw_ref, gbf_ref = read(os.path.join(path, "comparisons", "extraction_report.csv"), "Momega", "GBF_reference")
    for ax in axes:
        line, = ax.plot(mass*omega, gbf, label = rf"$\ell = {l}$")
        # ax.plot(Mw_ref, gbf_ref, "--", color = line.get_color(), alpha = 0.7, lw = 1, label = rf"ODE reference, $\ell = {l}$")
    summary += [(l, w, g) for w, g in zip(mass*omega, gbf)]

axes[0].set(xlabel = r"$M\omega$", ylabel = r"$\Gamma_\ell(M\omega)$", title = "Grey-body factors")
axes[1].set(xlabel = r"$M\omega$", ylabel = r"$\Gamma_\ell(M\omega)$", title = "Grey-body factors (log scale)", xscale = "log", yscale = "log")
for ax in axes:
    ax.grid(True, which = "both")
    ax.legend(fontsize = 8)
fig.tight_layout()
fig.savefig(os.path.join(multi_dir, "multi_l_GBF.png"), dpi = 200)

with open(os.path.join(multi_dir, "multi_l_GBF.csv"), "w", newline = "") as f:
    writer = csv.writer(f)
    writer.writerow(["l", "Momega", "PINN_GBF"])
    writer.writerows([[l, f"{w:.10e}", f"{g:.12e}"] for l, w, g in summary])
print(f"Saved combined GBF figure to {os.path.join(multi_dir, 'multi_l_GBF.png')}", flush = True)
PY
fi

echo "============================================================"
echo "Run complete."
echo "============================================================"