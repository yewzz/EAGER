# PyTorch Implementation of EAGER 

Code of paper *“EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration”* [(pdf)](https://arxiv.org/abs/2406.14017) accepted by KDD2024.



## Data preparation
- raw_data_file
  
The raw 5-core data can be downloaded from [official website](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).
- preprocess
  
Set *data_set_nome* (e.g. 'Amazon_Beauty') and *have_processed_data* to False. Run:
 ```
python train_rec.py
```
After it, set *have_processed_data* to True to avoid repeated preprocessing.

## Training

- DIN

To constuct the behavior code, you need to train a DIN model. Run:
```
python train_din.py
```

- EAGER

With the pre-trained DIN model, set *DIN_Model_path* to its checkpoint path. Run:
```
python train_rec.py
```

## Acknowledgements

We greatly appreciate the official RecForest [repository](https://github.com/wuchao-li/RecForest). Our code is built on their framework.
