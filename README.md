Python code to reproduce the experiments of our Neuroimage paper
[Multi-subject Multi-subject MEG/EEG source imaging with sparse multi-task regression](https://www.sciencedirect.com/science/article/pii/S1053811920303347)


1. To reproduce these experiments, the [Cam-CAN](https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/) and [DS117](https://www.openfmri.org/dataset/ds000117/) datasets
must be downloaded.

2. Modify the local repository urls in the `config.py` file accordingly.

3. To perform the benchmarked simulations, please install the Python package
provided in `sparse-mtr` by running `Python setup.py develop` from within
the repository.

4. All simulations can be performed by running `run_simulation.py`. Real data
experiments can be performed by running `run_real_data.py` and the scripts
`run_lasso` and `run_remwe` in `/real_data_fmri_mwe`.
