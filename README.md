# Deciphering Mutation Effects on Drug-Target Affinity Through a Label-Aware Contrastive Learning Framework

We propose a label-aware contrastive learning framework, **DeepMutDTA**, to decode the effects of mutations on drug-target affinity (DTA). Our approach introduces two significant contributions to DTA prediction, particularly concerning protein mutations. 
1. **DeepMutDTA** is a sequence-based model designed to predict DTA by leveraging FastFormer for processing protein sequences and MolFormer for drug SMILES. A Top-K attention mechanism is used to identify key drug-target interactions, enabling accurate predictions, especially in scenarios with limited experimental data.
2. We introduce **SimSiam-MuTF**, a label-aware contrastive learning fine-tuning framework that enhances DeepMutDTA's ability to handle protein mutations. By utilizing a supervised SimSiam network and a rank-based loss function, this framework boosts the model’s performance on mutation-specific data, even with minimal training samples.

# Data
```
data/platinum fold include the data of platinum dataset.
data/data_SimSiam_MuTF/GDSC fold include the data of GDSC dataset.
data/data_SimSiam_MuTF/PPI_1102 fold include the data of PPI_1102 dataset.
data/data_SimSiam_MuTF/PPI_1402 fold include the data of PPI_1402 dataset.
data/sars_cov_2.csv is the data of sars_cov_2 dataset.

Due to GitHub's limitations, the pre-training data is accessible via the following URL:
1. BindingDB: https://www.bindingdb.org/rwd/bind/index.jsp
2. BioLip: https://zhanggroup.org/BioLiP/
Or, you can download the filtered data on: https://drive.google.com/file/d/1-Edz8NUyAWJ48w71tq9miHtxnxKCjk-X/view?usp=sharing
```

# Requirements
```
torch 2.1.0
python 3.8.19
numpy 1.24.3
pandas 2.0.3
scikit-learn 0.24.0
scipy 1.10.1
transformers 4.40.1
```

# Pretrain model

1. The pretrained model is available at https://drive.google.com/file/d/1r5F9cnOgDpu85VYhsgvAuVtCTcTw3UQu/view?usp=sharing. You can download the model to the model folder or another folder and simply modify line 23 of main.py to specify the path.

2. The model of Molformer can be available on https://github.com/IBM/molformer. You can download the model the any folder and simply modify line 41 and 48 of utils.py and line 111 of model.py to specify the path.

# Run the demo

```
CUDA_VISIBLE_DEVICES=0 python main.py
```
If you want to load the pretrained model for fine-tuning, specify the path and use:
```
CUDA_VISIBLE_DEVICES=0 python main.py --load_model True
```
