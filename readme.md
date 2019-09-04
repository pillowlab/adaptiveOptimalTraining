# adaptiveOptimalTraining #

This package includes the matlab code for the paper:

- Ji Hyun Bak, Jung Yoon Choi, Athena Akrami, Ilana Witten, Jonathan Pillow. (2016) **Adaptive optimal training of animal behavior.**
_Advances in Neural Information Processing Systems 29._ [[link]](https://papers.nips.cc/paper/6344-adaptive-optimal-training-of-animal-behavior)


### Sample scripts:

`getSimDat.m`: generates a simulated behavior dataset (run this first!)
    - surrogate for a real animal behavior dataset.

`AOT_script_estWgt.m`: script for analyzing past observations,

- first with the random-walk prior only (hyperparameter sigma)
    - corresponds to Fig 2 in paper
- then with added learning component as drift (hyperparameter alpha)
    - corresponds to Fig 3 in paper

`AOT_script_training.m`: script for simulated active/passive training
    - corresponds to Fig 4 and S2 in paper


### Core functions:

* `funs_MNLogistic.m`: (this is a script)
    contains basic operations for (multinomial) logistic model
    usually called at the beginning of each core function

* `getMAP_RWprior.m`:
    does the MAP estimate for the weights with the random walk prior
    - getLP_MNLogistic_RWprior (core external subfunction)
    - negLogPost_MNLRW (wrapper for getLP)

* `getSimRat_active.m`:
    runs a simulated active training experiment
    - calls getPolGrad_discrimTask.m

* `getPolGrad_discrimTask.m`:
    calculates the policy gradient and the higher gradients,
    taylored for the specific task / model structure
