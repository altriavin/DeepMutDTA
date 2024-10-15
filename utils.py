import heapq
import os
from matplotlib.colors import ColorConverter
import numpy as np
import pandas as pd

import codecs
import torch
from subword_nmt.apply_bpe import BPE
from tape import TAPETokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from sklearn.metrics import mean_squared_error, roc_curve, auc, precision_recall_curve
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

def smiles_tokenizer(smiles):
    tokenizer = AutoTokenizer.from_pretrained("/root_path/MoLFormer", trust_remote_code=True)
    print("Tokenizing smiles...")
    smiles_token = tokenizer(smiles, return_tensors="pt")
    return smiles_token


def seqs_tokenizer(seqs):
    seq_token = TAPETokenizer(vocab='unirep')
    print(f'Tokenizing seqs...')
    seq_tokens = [list(seq_token.encode(seq)) for seq in tqdm(seqs)]
    return seq_tokens


def tokenizer_singal_smiles(smile):
    smiles_tokenizer = AutoTokenizer.from_pretrained("/root_path/MoLFormer", trust_remote_code=True)
    smile_token = smiles_tokenizer(smile)["input_ids"]
    decoded_tokens = smiles_tokenizer.convert_ids_to_tokens(smile_token)
    return decoded_tokens


def tokenizer_seq_smile(smiles, seqs_wt, seqs_mt):
    smiles_tokenizer = AutoTokenizer.from_pretrained("/root_path/MoLFormer", trust_remote_code=True)
    seq_tokenizer = TAPETokenizer(vocab='unirep')

    smiles_token, seqs_wt_token, seqs_mt_token = [], [], []

    print(f'Tokenizing smiles and seqs...')
    for idx, (smile, seq_wt) in enumerate(zip(smiles, seqs_wt)):
        smile_token = smiles_tokenizer(smile)["input_ids"]

        seq_wt_token = seq_tokenizer.encode(seq_wt)
        seq_mt_token = seq_tokenizer.encode(seqs_mt[idx])
        smiles_token.append(smile_token)
        seqs_wt_token.append(seq_wt_token)
        seqs_mt_token.append(seq_mt_token)
    return smiles_token, seqs_wt_token, seqs_mt_token


class CPIDataset(Dataset):
    def __init__(self, smiles, seqs_wt, labels_wt, seqs_mt, labels_mt):
        self.smiles = smiles
        self.seqs_wt = seqs_wt
        self.labels_wt = labels_wt
        self.seqs_mt = seqs_mt
        self.labels_mt = labels_mt
    
    def __len__(self):
        return len(self.labels_wt)

    def __getitem__(self, idx):
        return self.smiles[idx], self.seqs_wt[idx], self.labels_wt[idx], self.seqs_mt[idx], self.labels_mt[idx]


def padding_seq(smiles, max_smile, pad_value=0):
    smiles_padded, smiles_mask = [], []
    for s in smiles:
        mask = []
        if len(s) > max_smile:
            s = s[:max_smile]
            mask = [1] * len(s)
        else:
            mask = [1] * len(s) + [0] * (max_smile - len(s))
            s = list(s) + [pad_value] * (max_smile - len(s))
        smiles_padded.append(s)
        smiles_mask.append(mask)
    smiles_padded = torch.Tensor(np.array(smiles_padded)).long()
    smiles_mask = torch.Tensor(np.array(smiles_mask)).float()
    return smiles_padded, smiles_mask


def collate_fn(batch):
    smiles_token, seqs_wt_token, label_wt, seqs_mt_token, label_mt = zip(*batch)
    max_seq, max_smile = 1000, 100

    seqs_wt_padded, seqs_wt_mask = padding_seq(seqs_wt_token, max_seq, pad_value=0)
    seqs_mt_padded, seqs_mt_mask = padding_seq(seqs_mt_token, max_seq, pad_value=0)
    smiles_padded, smiles_mask = padding_seq(smiles_token, max_smile, pad_value=2)
    label_wt, label_mt = torch.tensor(np.array(label_wt)).float(), torch.tensor(np.array(label_mt)).float()
    
    return smiles_padded, smiles_mask, seqs_wt_padded, seqs_wt_mask, label_wt, seqs_mt_padded, seqs_mt_mask, label_mt


def get_data_loader(path, batch_size=256, shuffle=True):
    print(f'path: {path}')
    
    path = "/root_data_path/" + path + ".csv"
    data = pd.read_csv(path)
    if "gdsc" in path:
        smiles, seq_wt, seq_mt, label_wt, label_mt = data["drug_smile"].values, data["seq_wt"].values, data["seq_mt"].values, data["label_wt"].values, data["label_mt"].values
    else:
        smiles, seq_wt, seq_mt, label_wt, label_mt = data["smile"].values, data["seq_wt"].values, data["seq_mt"].values, data["label_wt"].values, data["label_mt"].values
    
    smiles_token, seqs_wt_token, seqs_mt_token = tokenizer_seq_smile(smiles, seq_wt, seq_mt)
    CPI_dataset = CPIDataset(smiles_token, seqs_wt_token, label_wt, seqs_mt_token, label_mt)
    data_loader = DataLoader(CPI_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return data_loader


def eval(test_targets, test_preds):
    test_targets = np.array(test_targets).flatten()
    test_preds = np.array(test_preds).flatten()
    mse = mean_squared_error(test_targets, test_preds)
    rmse = np.sqrt(mse)
    
    pcc, p_value = pearsonr(test_targets, test_preds)
    scc, s_p_value = spearmanr(test_targets, test_preds)

    conindex = concordance_index(test_targets, test_preds)
    return {
        "RMSE": rmse,
        'PCC': pcc,
        'SCC': scc,
        'conindex': conindex
    }