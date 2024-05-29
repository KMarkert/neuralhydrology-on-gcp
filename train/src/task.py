import pickle
from pathlib import Path
import subprocess

import torch

from cloudpathlib import AnyPath
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run
from neuralhydrology.utils.config import Config

# by default we assume that you have at least one CUDA-capable NVIDIA GPU
if torch.cuda.is_available():
    start_run(config_file=Path("basin.yml"))

# fall back to CPU-only mode
else:
    start_run(config_file=Path("basin.yml"), gpu=-1)


# evaluate the trained modle in the task
run_dir = list(Path("runs/").glob('**'))[1]
eval_run(run_dir=run_dir, period="test")

with open(run_dir / "test" / "model_epoch050" / "test_results.p", "rb") as fp:
    results = pickle.load(fp)

for basin in results.keys():
    # extract observations and simulations
    qobs = results[basin]['1D']['xr']['QObs(mm/d)_obs']
    qsim = results[basin]['1D']['xr']['QObs(mm/d)_sim']
    print(f'Evaluating basin: {basin}')
    values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
    for key, val in values.items():
        print(f"{key}: {val:.3f}")
    
    print('\n')


# write the output model to cloud storage for later use
model_outdir = AnyPath('gs://neuralhydro')

# transfer the contents of run directory to cloud storage bucket
for p in run_dir.rglob("*"):
    # copy a cubdir if there is one
    if p.is_dir():
        subdir = model_outdir / p
        if not subdir.exists():
            subdir.mkdir(parents=True)
    # simply write the bytes to the path on cloud storage
    else:
        name = p.name
        parent = p.parent
        gcs_path = model_outdir/ parent / name
        gcs_path.write_bytes(p.read_bytes())

