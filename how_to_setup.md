# 0. Connect to the server 
Set up SSH keys, set up ssh configs etc.

https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/setup_ssh/


# 1. Set up the virtual environment (on Login node)
You only need to do this once.
```
source ./llm_env/bin/setup.sh
sh ./llm_env/install_requirements.sh
```
# 2. Download the data (on Login node)
There are three datasets:
- `openwebtext`
- `shakespeare_char`
- `tiny_shakespeare`
We will use `shakespeare_char` for the experiments.
You can download the data by running the following command:
```
python data/shakespeare_char/prepare.py
```


# 3. Connect to the compute node (on Login node)
### Connect to the normal compute node (max 24 hours)
`srun --time=4:00:00 --cpu_bind=none --nodes=1 --ntasks=1 --partition=dc-gpu -A training2520 --pty /bin/bash`

### Connect to the debug compute node (max 2 hours)
`srun --time=2:00:00 --cpu_bind=none --nodes=1 --ntasks=1 --partition=dc-gpu-devel -A training2520 --pty /bin/bash`

### Connect to the reserved compute node
`srun --time=4:00:00 --cpu_bind=none --nodes=1 --ntasks=1 --partition=dc-gpu -A training2520 --reservation=training2520 --pty /bin/bash`

Once you connect to the compute node, you should be able to see 4 GPUs via `nvidia-smi`.


# (Optional)  Activate your virtual environment
Everytime you connect to the server, or submit the job via `sbatch`, you need to activate the virtual environment.
```bash
source ./llm_env/bin/activate
```


# 4. Run the experiments (on compute node)
For `MuP`
```bash
sh mup_examples/mutransfer_lr_shakespeare_char/mup/run.sh
```

For `SP`
```bash
sh mup_examples/mutransfer_lr_shakespeare_char/sp/run.sh
```

To see the results, you can plot them via Jupyter Notebook.
`mup_examples/mutransfer_lr_shakespeare_char/short_run_plot.ipynb`


