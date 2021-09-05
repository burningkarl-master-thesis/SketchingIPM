# SketchingIPM
An interior-point method for LPs using sketching techniques

This repository contains the source code related to the numerical experiments described in the "Sketching for Infeasible Interior-Point Methods" paper.
The code is provided to help others implementing sketching techniques inside SciPy's IPM implementation and is not intended as a way to replicate the experiments reported in the paper.

The repository contains two main folders:
The `experiments` folder contains code to run the experiments and record their results, while the `graphs` folder contains code that generates visualisations of those results.


## Experiments

To run the experiments, you will first need to install the dependencies.
Inside the `experiments` folder, run

```bash
pipenv sync
pipenv shell
```

to setup a virtual environment with all the necessary dependencies and activate it.

Because using sketching techniques inside SciPy's IPM required me to edit the SciPy source files, I recorded my changes in a `scipy` fork.
Please clone it, checkout the `sketching-for-ipm` branch, find all the files with changes using `git diff v1.6.2 --stat` and overwrite their counterparts inside the `lib/python3.8/site-packages/scipy` subfolder of the virtual environment.
You can find the path to the virtual environment using `pipenv --venv`.
While this process is cumbersome, it avoids recompiling SciPy from source, which would be required if one uses `pipenv install` with the forked GitHub repository.

To record the experiment results I used `wandb`, which is free for personal accounts.
You will need to login with `wandb login` before you can start tracking your experiments.

After all these setup steps, you can run
```bash
python ipm.py --config testconfig.json
```
to perform an experiment with the configuration options from `testconfig.json`.

I tried my best to choose sensible parameter names and write self-explanatory code, but given the lack of comments, I'm happy to answer any questions via GitHub issues.


## Graphs

The code inside the `graphs` directory was used to generate graphs that visualise the results from the numerical experiments for the paper.
It was written in an ad-hoc fashion and will require some changes to be useful for other applications.
In principle, the dependencies can be installed as above using `pipenv sync && pipenv shell` and the scripts can be executed by

```bash
python plot_sketching_parameters.py w_and_s_ipm_7
```

for example (This should generate the sketching parameters graphs from the data inside the `w_and_s_ipm_7` group).


## Acknowledgements

I am deeply thankful for the hard work of the SciPy contributors.
Their largely well-documented code helped me to implement a modified interior-point method without reinventing the wheel.
