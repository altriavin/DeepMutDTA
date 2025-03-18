
---

# Label-Aware Contrastive Learning Enables Drug Discovery for Resistance Mutation Targets

**DeepMutDTA** is a novel label-aware contrastive learning framework that deciphers the impact of protein mutations on drug-target affinity (DTA). This repository provides a comprehensive implementation designed to predict DTA while addressing challenges related to mutation-induced variations. By integrating advanced sequence-based models with a specialized fine-tuning mechanism, DeepMutDTA achieves robust performance even when experimental data are limited.

---

## Overview

DeepMutDTA makes two significant contributions to the field of DTA prediction, particularly concerning protein mutations:

1. **Sequence-Based DTA Prediction:**  
   - **Protein and Drug Encoding:** Utilizes *FastFormer* for efficient processing of protein sequences and *MolFormer* for encoding drug SMILES.  
   - **Top-K Attention Mechanism:** Identifies key interactions between drugs and target proteins, which is critical for accurate affinity predictions, especially in data-scarce scenarios.

2. **Enhanced Mutation-Specific Prediction with SimSiam-MuTF:**  
   - **Label-Aware Contrastive Learning:** Implements a supervised SimSiam network combined with a rank-based loss function to better capture mutation-specific features.  
   - **Robust Fine-Tuning:** Improves model performance on mutation-specific data even with minimal training samples by focusing on the differences induced by mutations.

---

## Pre-trained Models

To accelerate your experiments, pre-trained models are provided:

- **DeepMutDTA Pre-trained Model:**  
  Download the model from [this link](https://drive.google.com/file/d/1r5F9cnOgDpu85VYhsgvAuVtCTcTw3UQu/view?usp=sharing).  
  Place the downloaded file in the `model` folder (or another directory of your choice) and update the file path on **line 23 of `main.py`** accordingly.

- **MolFormer Model:**  
  The MolFormer model is available from the [IBM MolFormer GitHub repository](https://github.com/IBM/molformer).  
  Download this model to your desired directory and modify the file paths on:  
  - **Lines 41 and 48 of `utils.py`**  
  - **Line 111 of `model.py`**

---

## Datasets

The repository supports both demo datasets and custom data. Below are the details for each option:

### 1. Training and Testing on Demo Data

You can access various public datasets used in our experiments:

| **Dataset**   | **URL** |
| ------------- | ------- |
| **BindingDB** | [BindingDB](https://www.bindingdb.org/rwd/bind/index.jsp) |
| **BioLip**    | [BioLip](https://zhanggroup.org/BioLiP/index.cgi) |
| **DEKOIS2.0** | [DEKOIS2.0](http://www.dekois.com) |
| **DUD-E**     | [DUD-E](https://dude.docking.org/) |
| **Platinum**  | [Platinum](https://biosig.lab.uq.edu.au/platinum) |
| **SARS-CoV-2**| Derived from literature |
| **HIV**       | [HIV Database](https://hivdb.stanford.edu/) |
| **TTD**       | [TTD](https://idrblab.net/ttd/ttd-search/mutation) |
| **COSMIC**    | [COSMIC](https://cancer.sanger.ac.uk/cosmic) |

Alternatively, you may use our preprocessed data available at:

| **Dataset**       | **URL** |
| ----------------- | ------- |
| **Pre-training**  | [pre-training dataset](https://drive.google.com/file/d/1-Edz8NUyAWJ48w71tq9miHtxnxKCjk-X/view?usp=sharing) |
| **DEKOIS2.0**     | [DEKOIS2.0](https://drive.google.com/drive/folders/14Dn_Y4eq3ygLUecaqWTStFWlZS2N6xBc?usp=sharing) |
| **DUD-E**         | [DUD-E](https://drive.google.com/drive/folders/18yn-Zt1x-2nxzL3dubRTci8GPzV2Jmns?usp=sharing) |
| **Mutation Data** | [Mutation data](https://drive.google.com/drive/folders/127SAD1cS_xInSdPO1V15vIZu1o6h8PXi?usp=sharing) |
| **PPI Data**      | [PPI dataset](https://drive.google.com/open?id=12VaSAVw1q_8N2YRTPMBLmvlDZeXDqTfm&usp=drive_fs) |

### 2. Training and Testing on Custom Data

If you wish to apply DeepMutDTA to your own datasets, please ensure your data is preprocessed into one of the following formats:

#### Regression Task Format

For regression tasks, format your data as follows:

| **seq_wt**      | **seq_mt**      | **smile**    | **label_wt**                         | **label_mt**                         |
| --------------- | --------------- | ------------ | ------------------------------------- | ------------------------------------- |
| wt_protein_seq  | mt_protein_seq  | drug_smile   | -log( wt_affinity / 1e9 )             | -log( mt_affinity / 1e9 )             |

#### Classification Task Format

For classification tasks, format your data as follows:

| **seq_wt**      | **seq_mt**      | **smile**    | **label_wt** | **label_mt** |
| --------------- | --------------- | ------------ | ------------ | ------------ |
| wt_protein_seq  | mt_protein_seq  | drug_smile   | 0            | 1            |

Ensure your dataset follows the above specifications before starting the training process.

---

## Requirements

Before running the code, please ensure that you have installed the following packages:

```bash
torch==2.1.0  
python==3.8.19  
numpy==1.24.3  
pandas==2.0.3  
scikit-learn==0.24.0  
scipy==1.10.1  
transformers==4.40.1  
```

You can install these dependencies using `pip`:

```bash
pip install torch==2.1.0 python==3.8.19 numpy==1.24.3 pandas==2.0.3 scikit-learn==0.24.0 scipy==1.10.1 transformers==4.40.1
```

---

## Usage

### 1. Training and Testing on Demo Data

To run the model using the demo datasets, execute the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --load_model False --save_model False --alpha 0.5 --beta 0.4 --learn_rate 5e-5 --batch_size 32 --epochs 100 --task_type clf
```

- **Regression Task:** Set `--task_type reg` if you want to test the model for regression.
- **Classification Task:** Set `--task_type clf` if you want to test the model for classification.

### 2. Running on Custom Data

After formatting your custom dataset according to the provided guidelines, update the corresponding file paths in the configuration files or in the `main.py` script. Then, run the training script with your desired parameters.

---

## Acknowledgements

This project builds upon state-of-the-art architectures including FastFormer, MolFormer, the SimSiam network, RnC loss function and SCL loss function. We gratefully acknowledge the contributions from the respective developers and the research community that made these advancements possible.

---

## Contract

If you have any questions, feel free to contact me by email: vinaltria@csu.edu.cn

---
