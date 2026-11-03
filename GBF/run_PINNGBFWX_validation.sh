#!/bin/bash
set -euo pipefail

# Wrapper for:
#   1. NumericalGBF.py (optional; writes ./Numerical/l{mode}/Output/numericalGBF.csv)
#   2. the PINN script 
#
# All options are passed as named flags (--mass 0.5 --mode 2 ...) and forwarded unchanged to the PINN.
# The shared ones (mass, mode, x_extract, omega_start, omega_final, query_steps) are also given to NumericalGBF.py.
# Wrapper-only option:
#   --run_numerical yes|no   yes (default): run NumericalGBF.py first and compare the PINN against its CSV
#                            no: skip it and pass --no_numerical_csv to the PINN
#
# Environment overrides: PYTHON_BIN, PINN_SCRIPT, NUMERICAL_SCRIPT, REQUIRE_GPU (1 = refuse to run on CPU).

# Run from the project directory containing this wrapper.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Change this if you need a specific Conda/Python executable.
PYTHON_BIN="${PYTHON_BIN:-/home/2665592k/.conda/envs/pinns-gpu/bin/python}"
PINN_SCRIPT="${PINN_SCRIPT:-PINNGBFWX.py}"
NUMERICAL_SCRIPT="${NUMERICAL_SCRIPT:-NumericalGBF.py}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"

echo "Python: $PYTHON_BIN"

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

# The PINN imports gbf_reference.py from its own directory
for required in "$PINN_SCRIPT" gbf_reference.py; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: $required not found in $SCRIPT_DIR" >&2
        exit 2
    fi
done

# -----------------------------
# Parse named options
# -----------------------------
RUN_NUMERICAL="yes"
MASS=""
MODE=""
X_EXTRACT=""
OMEGA_START="1.2"      # PINN defaults, used if the flag is not given
OMEGA_FINAL="0.01"
QUERY_STEPS="200"
PINN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_numerical) RUN_NUMERICAL="$2"; shift 2; continue ;;
        --mass)          MASS="$2" ;;
        --mode)          MODE="$2" ;;
        --x_extract)     X_EXTRACT="$2" ;;
        --omega_start)   OMEGA_START="$2" ;;
        --omega_final)   OMEGA_FINAL="$2" ;;
        --query_steps)   QUERY_STEPS="$2" ;;
    esac
    PINN_ARGS+=("$1")
    shift
done

if [[ -z "$MASS" || -z "$MODE" || -z "$X_EXTRACT" ]]; then
    echo "Usage: $0 --mass M --mode L --x_extract X [any other PINNGBFWX.py options] [--run_numerical yes|no]" >&2
    exit 2
fi

if [[ "$RUN_NUMERICAL" == "yes" ]]; then
    echo "============================================================"
    echo "Starting $NUMERICAL_SCRIPT on the dense query frequency grid"
    echo "query_steps=${QUERY_STEPS}"
    echo "============================================================"

    "$PYTHON_BIN" "$NUMERICAL_SCRIPT" \
        --mass "$MASS" \
        --mode "$MODE" \
        --x_extract "$X_EXTRACT" \
        --omega_start "$OMEGA_START" \
        --omega_final "$OMEGA_FINAL" \
        --num_steps "$QUERY_STEPS"

    echo "Numerical solver finished successfully."
elif [[ "$RUN_NUMERICAL" == "no" ]]; then
    echo "Skipping $NUMERICAL_SCRIPT (--run_numerical no)."
    PINN_ARGS+=(--no_numerical_csv)
else
    echo "ERROR: --run_numerical must be yes or no (got '$RUN_NUMERICAL')" >&2
    exit 2
fi

echo "============================================================"
echo "Starting $PINN_SCRIPT ${PINN_ARGS[*]}"
echo "============================================================"

"$PYTHON_BIN" "$PINN_SCRIPT" "${PINN_ARGS[@]}"

echo "============================================================"
echo "Validation run complete."
echo "============================================================"
