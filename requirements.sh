# All of these commands install a CPU version of torch for some reason, so I used the below conda command to install pytorch-gpu

#pip install torch==1.13.0+cu117 torchvision==0.10.0+cu117 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
#conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
#conda install -y pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 cudatoolkit=11.3 -c pytorch

pip install weaver-core

#conda install pytorch-gpu -c pytorch # this works
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

pip install seaborn
pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html  # I had to install this one on lxplus as fcc machine was using an old gcc version
pip install scipy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install tqdm
pip install PyYAML
pip install awkward0
pip install uproot
pip install awkward
pip install vector
pip install lz4
pip install xxhash
pip install tables
pip install tensorboard
pip install wandb
#pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

conda install -y cudatoolkit=11.3