This package includes the matlab code for the NIPS 2016 paper,
Adaptive optimal training of animal behavior.

2016-2017 Ji Hyun Bak.

(Last update 4/7/2017)

-------------------------------------------------------------------------
Sample scripts:
-------------------------------------------------------------------------

getSimDat: generates a simulated behavior dataset (run this first!)
    - surrogate for a real animal behavior dataset.

AOT_script_estWgt: script for analyzing past observations
- first with the random-walk prior only (hyperparameter sigma)
    - corresponds to Fig 2 in paper
- then with added learning component as drift (hyperparameter alpha)
    - corresponds to Fig 3 in paper

AOT_script_training: script for simulated active/passive training
    - corresponds to Fig 4 and S2 in paper

-------------------------------------------------------------------------
Core functions:
-------------------------------------------------------------------------

* funs_MNLogistic: (this is a script)
    contains basic operations for (multinomial) logistic model
    usually called at the beginning of each core function

* getMAP_RWprior:
    does the MAP estimate for the weights with the random walk prior
    - getLP_MNLogistic_RWprior (core external subfunction)
    - negLogPost_MNLRW (wrapper for getLP)

* getSimRat_active:
    runs a simulated active training experiment
    - calls getPolGrad_discrimTask.m

* getPolGrad_discrimTask:
    calculates the policy gradient and the higher gradients,
    taylored for the specific task / model structure
