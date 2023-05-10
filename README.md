# MLPF FCC
particle-flow with ml 


## ML pipeline:
- The dataloaders, train scripts and tools are currently based on [Weaver](https://github.com/hqucms/weaver-core/tree/main), the reason for this is that we are importing a root file that contains the dataset and these files can be large. Weaver has all the tools to read and load from the rootfile and also develops and iterable dataloader that prefetches some data. Currently this dataset includes events. One event is formed by hits (which can be tracks or calo hits). An input is an event in the form of a graph, and the output is a single particle (in coming versions of the dataset there will be more). 
- Models: The goal of the current taks is to regress the particle's information (coordinates and energy). Currently the best approach is the [object condensation]([https://link-url-here.org](https://github.com/hqucms/weaver-core/tree/main](https://arxiv.org/abs/2002.03605), since it allows to regress a variable number of particles. 
- Training: To train a model run the following command 
`python -m src.train --data-train /eos/user/m/mgarciam/datasets/pflow/tree_mlpf2.root --data-config config_files/config_2_newlinks.yaml --network-config src/models/wrapper/example_gravnet_model.py --model-prefix models_trained/ --num-workers 0 --gpus --batch-size 100 --start-lr 1e-3 --num-epochs 1000 --optimizer ranger --fetch-step 1 --log logs/train.log --log-wandb --wandb-displayname test --wandb-projectname mlpf --condensation`

##Visualization 
Runs for this project can be found in the following work space:[wandb](https://github.com/hqucms/weaver-core/tree/main](https://arxiv.org/abs/2002.03605](https://wandb.ai/imdea_dolo/mlpf?workspace=user-imdea_dolo) 

## Envirorment 
To set up the env create a conda env following the instructions from [Weaver](https://github.com/hqucms/weaver-core/tree/main) and also install the packages in the requirements.sh script above 

