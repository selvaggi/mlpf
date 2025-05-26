# MLPF FCC
Machine learning based pipeline for particle flow at FCC. 

Latest documentation/ presentations can be found here:
- https://repository.cern/records/n9wc2-09n03
- https://indico.cern.ch/event/1408515/contributions/6521312/

Additional information and setup instructions can be found in the wiki!


## ML pipeline:
- The dataloaders, train scripts and tools are currently based on [Weaver](https://github.com/hqucms/weaver-core/tree/main), the reason for this is that we are importing a root file that contains the dataset and these files can be large. Weaver has all the tools to read and load from the rootfile and also develops and iterable dataloader that prefetches some data. Currently this dataset includes events. One event is formed by hits (which can be tracks or calo hits). An input is an event in the form of a graph, and the output is a single particle (in coming versions of the dataset there will be more). 
- Models: The goal of the current taks is to regress the particle's information (coordinates and energy). Currently the best approach is the [object condensation](https://arxiv.org/abs/2002.03605), since it allows to regress a variable number of particles. 
- The pipeline includes the following four steps: training the clustering, training energy correction and PID, evaluation, plotting 
- Training: To train a model check the wiki/Training section 


## Visualization 
Runs for this project can be found in the following work space: https://wandb.ai/imdea_dolo/mlpf?workspace=user-imdea_dolo

## Environment 
You can use the docker container ```docker://dologarcia/gatr:v9``` and use singularity to run the container as:
 ```export APPTAINER_CACHEDIR=/home/ADDUSER/cache/```
 ```singularity  shell   -B /eos -B /afs  --nv docker://dologarcia/gatr:v9```

You might need to set up a conda env to run notebooks. To set up the env create a conda env following the instructions from [Weaver](https://github.com/hqucms/weaver-core/tree/main) and also install the packages in the requirements.sh script above 
Alternatively, you can try to use a pre-built environment from [this link](https://cernbox.cern.ch/s/Rwz2S35BUePbwG4) - the .tar.gz file was built using conda-pack on fcc-gpu-04v2.cern.ch.

## Energy correction

Firstly a simple neural network is trained to perform energy correction. We train two models separately for neutral and charged particles.

```python notebooks/13_NNs.py --prefix /eos/user/g/gkrzmanc/2024/EC_basic_model_charged --wandb_name NN_EC_train_charged --loss default --PIDs 211,-211,2212,-2212,11 --dataset-path /eos/user/g/gkrzmanc/2024/ft_ec_saved_f_230424/cluster_features/ --batch-size 8 --corrected-energy --gnn-features-placeholders 32```

```python notebooks/13_NNs.py --prefix /eos/user/g/gkrzmanc/2024/EC_basic_model_neutral --wandb_name NN_EC_train_neutral --loss default --PIDs 130,2112,22            --dataset-path /eos/user/g/gkrzmanc/2024/ft_ec_saved_f_230424/cluster_features/ --batch-size 8 --corrected-energy --gnn-features-placeholders 32```

The produced models are then loaded in `src/models/GATr/Gatr_pf_e.py`.

Evaluation / further training: `--ec-model gat-concat` or `--ec-model dnn`.
