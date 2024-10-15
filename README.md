# Deciphering Mutation Effects on Drug-Target Affinity Using a Supervised SimSiam Network

DeepMutDTA introduces two key contributions to drug-target affinity (DTA) prediction, particularly in the context of protein mutations. First, we developed DeepMutDTA, a sequence-based model for predicting DTA. It uses FastFormer for protein sequences and MolFormer for drug SMILES, with a Top-K attention mechanism to capture key interactions. This allows for accurate predictions of DTA, especially when experimental data is limited. Second, we propose SimSiam-MuTF, a fine-tuning framework that improves DeepMutDTA’s ability to handle protein mutations. Using a supervised SimSiam network and a rank-based loss function, this framework enhances the model’s performance on mutation data, even with few training samples.

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

# Run the demo

```
CUDA_VISIBLE_DEVICES=0 python main.py
```
