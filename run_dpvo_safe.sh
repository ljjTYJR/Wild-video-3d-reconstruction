#!/bin/bash

# Script to run DPVO with Arrow cleanup workaround

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate droid

# Set environment variables to prevent Arrow cleanup issues
export ARROW_PRE_0_15_IPC_FORMAT=1
export PYARROW_IGNORE_TIMEZONE=1
export OMP_NUM_THREADS=1

# Disable Python buffering for better error visibility
export PYTHONUNBUFFERED=1

# Run DPVO with all arguments passed to this script
python dpvo_demo.py "$@"

# Exit code handling
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "DPVO exited with code: $exit_code"
fi

exit $exit_code