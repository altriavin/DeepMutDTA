# Deciphering Mutation Effects on Drug-Target Affinity Through a Label-Aware Contrastive Learning Framework

We propose a label-aware contrastive learning framework, **DeepMutDTA**, to decode the effects of mutations on drug-target affinity (DTA). Our approach introduces two significant contributions to DTA prediction, particularly concerning protein mutations. 
1. **DeepMutDTA** is a sequence-based model designed to predict DTA by leveraging FastFormer for processing protein sequences and MolFormer for drug SMILES. A Top-K attention mechanism is used to identify key drug-target interactions, enabling accurate predictions, especially in scenarios with limited experimental data.
2. We introduce **SimSiam-MuTF**, a label-aware contrastive learning fine-tuning framework that enhances DeepMutDTA's ability to handle protein mutations. By utilizing a supervised SimSiam network and a rank-based loss function, this framework boosts the model’s performance on mutation-specific data, even with minimal training samples.

# Data
```
data/platinum fold include the data of platinum dataset.
data/GDSC fold include the data of GDSC dataset.
data/PPI_1102 fold include the data of PPI_1102 dataset.
data/PPI_1402 fold include the data of PPI_1402 dataset.

Due to GitHub's limitations, the pre-training data is accessible via the following URL:
1. BindingDB: https://www.bindingdb.org/rwd/bind/index.jsp
2. BioLip: https://zhanggroup.org/BioLiP/
```

# Requirements
```
torch 2.1.0
python 3.8.19
numpy 1.24.3
pandas 2.0.3
scikit-learn 0.24.0
scipy 1.10.1
subword-nmt 0.3.8
transformers 4.40.1
```

# Pretrain model
、、、
The pretrain model can be available on https://drive.google.com/file/d/1r5F9cnOgDpu85VYhsgvAuVtCTcTw3UQu/view?usp=sharing
、、、

# Run the demo

```
CUDA_VISIBLE_DEVICES=0 python main.py
```
