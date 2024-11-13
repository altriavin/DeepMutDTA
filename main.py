import heapq
import pickle

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from model_finetune import *
from utils import *
from loss_func import *

from tqdm import tqdm
from utils import get_data_loader, eval
from model import TransformerModel
import torch.nn as nn
import torch
import argparse
from model_finetune import SuperviseSimSiam


def train_test(train_loader, test_loader, val_loader, args, load_model=False):
    backbone_model = TransformerModel(args=args)
    if load_model == True:
        path = 'model/pretrain_model.pth'
        backbone_model.load_state_dict(torch.load(path))
        print(f'load backbone_model from {path}')
    else:   
        print(f'donot load pretrain_model!!!')
    backbone_model = backbone_model.cuda()
    model = SuperviseSimSiam(backbone_model=backbone_model, args=args)

    optimizer = torch.optim.Adam([ {'params': model.parameters(), 'lr': args.learn_rate}])
    
    mse_loss = nn.MSELoss()
    RnC_loss = RnCLoss(temperature=args.temp, label_diff='l1', feature_sim='l2')
    best_pcc, best_scc = 0.0, 0.0
    best_result = {}
    for epoch in range(args.epochs):
        model.train()
        for idx, (smiles_padded, smiles_mask, seqs_wt_padded, seqs_wt_mask, label_wt, seqs_mt_padded, seqs_mt_mask, label_mt) in enumerate(tqdm(train_loader, desc="Training...")):
            
            smiles_padded, smiles_mask, seqs_wt_padded, seqs_wt_mask, label_wt, seqs_mt_padded, seqs_mt_mask, label_mt = smiles_padded.cuda(), smiles_mask.cuda(), seqs_wt_padded.cuda(), seqs_wt_mask.cuda(), label_wt.cuda(), seqs_mt_padded.cuda(), seqs_mt_mask.cuda(), label_mt.cuda()
            emb_view_1_wt, emb_view_2_wt, emb_view_1_mt, emb_view_2_mt, score_wt, score_mt = model(smiles_padded, smiles_mask, seqs_wt_padded, seqs_wt_mask, seqs_mt_padded, seqs_mt_mask)
            
            rnc_loss_1 = RnC_loss(emb_view_1_wt, emb_view_2_mt, label_wt, label_mt)
            mse_loss_1 = mse_loss(score_wt.squeeze(), label_wt)
            rnc_loss_2 = RnC_loss(emb_view_2_wt, emb_view_1_mt, label_wt, label_mt)
            mse_loss_2 = mse_loss(score_mt.squeeze(), label_mt)
            loss = args.alpha * (rnc_loss_1 + rnc_loss_2) + args.beta * (mse_loss_1 + mse_loss_2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_result = test_model(model, test_loader)
        
        if test_result["PCC"] > best_pcc and test_result["SCC"] > best_scc:
            best_pcc, best_scc = test_result["PCC"], test_result["SCC"]

            val_result = test_model(model, val_loader)
            best_result = val_result

            print(f'epoch: {epoch}; pcc: {val_result["PCC"]}; scc: {val_result["SCC"]}')
            
            if args.save_model:
                torch.save(model.backbone.state_dict(), f'fintune_model.pth')
                print(f'save model to fintune_model.pth')
            else:
                print(f'donot save model!!!')
    return best_result


def test_model(model, data_loader):
    model.eval()
    preds_wt, actuals_wt, preds_mt, actuals_mt = [], [], [], []
    with torch.no_grad():
        for idx, (smiles_padded, smiles_mask, seqs_wt_padded, seqs_wt_mask, label_wt, seqs_mt_padded, seqs_mt_mask, label_mt) in enumerate(tqdm(data_loader, desc="Testing...")):
            smiles_padded, smiles_mask, seqs_wt_padded, seqs_wt_mask, seqs_mt_padded, seqs_mt_mask = smiles_padded.cuda(), smiles_mask.cuda(), seqs_wt_padded.cuda(), seqs_wt_mask.cuda(), seqs_mt_padded.cuda(), seqs_mt_mask.cuda()
            _, _, _, _, score_wt, score_mt = model(smiles_padded, smiles_mask, seqs_wt_padded, seqs_wt_mask, seqs_mt_padded, seqs_mt_mask)
            preds_wt.extend(score_wt.detach().cpu().numpy().tolist())
            actuals_wt.extend(label_wt.numpy().tolist())
            preds_mt.extend(score_mt.detach().cpu().numpy().tolist())
            actuals_mt.extend(label_mt.numpy().tolist())
    preds, actuals = preds_wt + preds_mt, actuals_wt + actuals_mt
    preds, actuals = np.array(preds).flatten().tolist(), np.array(actuals).flatten().tolist()
    result = eval(actuals, preds)
    return result


def cv_platinum(args):
    best_result = []
    for i in range(5):
        args.idx = i
        train_path, test_path, val_path = args.dataset + "_train_" + str(i), args.dataset + "_test_" + str(i), args.dataset + "_indepent"
        train_data_loader, test_data_loader = get_data_loader(path=train_path, batch_size=args.batch_size), get_data_loader(path=test_path, batch_size=args.batch_size)
        val_data_loader = get_data_loader(path=val_path, batch_size=args.batch_size)
        result = train_test(train_data_loader, test_data_loader, val_data_loader, args, load_model=args.load_model)
        best_result.append(result)
    print(f'best_result: {best_result}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='platinum', help='dataset name: platinum/gdsc')
    parser.add_argument('--hidden_act', default='gelu')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--learn_rate', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2)
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layer used')
    parser.add_argument('--hidden_size', type=int, default=512, help='the number of layer used')
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--vocab_size', type=int, default=31)
    parser.add_argument('--max_seq_len', type=int, default=1000)
    parser.add_argument('--max_drug_len', type=int, default=100)
    parser.add_argument('--intermediate_size', type=int, default=256)
    parser.add_argument('--topk', type=int, default=32)
    parser.add_argument('--out_dim', type=int, default=1024)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--temp', type=float, default=3.0)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--load_model', type=bool, default=False)

    args = parser.parse_args()

    cv_platinum(args=args)
    