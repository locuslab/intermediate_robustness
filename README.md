# Robustness between the worst and average case 

*A repository that implements intermediate robustness training and evaluation from the NeurIPS 2021 paper [Robustness between the worst and average case](https://proceedings.neurips.cc/paper/2021/file/ea4c796cccfc3899b5f9ae2874237c20-Paper.pdf). 
Created by [Leslie Rice](https://leslierice1.github.io/), [Anna Bair](https://annaebair.github.io/), [Huan Zhang](https://www.huan-zhang.com/) and [Zico Kolter](http://zicokolter.com).*

## Installation and usage
- To install all required packages run: `pip install -r requirements.txt`.
- Pretrained model weights can be downloaded [here](https://drive.google.com/drive/folders/1YCFXzdx2dGjmQGU30v6CRhUORjQsjHKV?usp=sharing). 
- To train (l_infty perturbations), run `python train.py -c {path_to_training_config_file}.json`. 
- To evaluate (l_infty perturbations), run `python train.py -c {path_to_evaluation_config_file}.json`. 
- To train (spatial transformations), run `python train_discrete.py -c {path_to_training_config_file}.json`.  
- To evaluate (spatial transformations), run `python eval_discrete.py --checkpoint {path_to_model_checkpoint}.pth`.  
