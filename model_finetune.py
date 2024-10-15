import torch.nn as nn

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_size=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            # print(f'x.shape: {x.shape}')
            x = self.layer2(x)
            # print(f'x.shape: {x.shape}')
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_size=512, out_dim=2048): # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SuperviseSimSiam(nn.Module):
    def __init__(self, args, backbone_model):
        super(SuperviseSimSiam, self).__init__()
        self.backbone = backbone_model.cuda()
        self.projection = projection_MLP(in_dim=args.hidden_size * 2, hidden_size=args.out_dim, out_dim=args.hidden_size * 2).cuda()
        self.predictor = prediction_MLP(in_dim=args.hidden_size * 2, hidden_size=args.out_dim, out_dim=args.hidden_size * 2).cuda()

        self.Sigma = nn.Sigmoid().cuda()

    def get_wt_encoder(self, seqs_padded_wt, seqs_masked_wt, smiles_padded, smiles_mask):
        emb_view_1, score, _, _ = self.backbone(smiles_padded, seqs_padded_wt, smiles_mask, seqs_masked_wt)
        # print(f'emb_view_1.shape: {emb_view_1.shape}')
        emb_view_1 = self.projection(emb_view_1)
        emb_view_2 = self.predictor(emb_view_1)
        return emb_view_1, emb_view_2, score
    
    def get_mt_encoder(self, seqs_padded_mt, seqs_masked_mt, smiles_padded, smiles_mask):
        emb_view_1, score, _, _ = self.backbone(smiles_padded, seqs_padded_mt, smiles_mask, seqs_masked_mt)
        emb_view_1 = self.projection(emb_view_1)
        emb_view_2 = self.predictor(emb_view_1)
        return emb_view_1, emb_view_2, score
    
    def get_case(self, smiles_padded, smiles_mask, seqs_padded, seqs_mask):
        emb, score = self.backbone(smiles_padded, seqs_padded, smiles_mask, seqs_mask)
        return emb, score
    
    def get_DDG_score(self, smiles_padded, smiles_mask, seqs_wt_padded, seqs_wt_mask, seqs_mt_padded, seqs_mt_mask):
        emb_wt, score_wt = self.backbone(smiles_padded, seqs_wt_padded, smiles_mask, seqs_wt_mask)
        emb_mt, score_mt = self.backbone(smiles_padded, seqs_mt_padded, smiles_mask, seqs_mt_mask)
        return score_mt - score_wt
    
    def get_output(self, smiles_padded, smiles_mask, seqs_padded, seqs_mask):
        emb, score = self.backbone(smiles_padded, seqs_padded, smiles_mask, seqs_mask)
        return emb, score
    
    def forward(self, smiles_padded, smiles_mask, seqs_wt_padded, seqs_wt_mask, seqs_mt_padded, seqs_mt_mask):
        emb_view_1_wt, emb_view_2_wt, score_wt = self.get_wt_encoder(seqs_wt_padded, seqs_wt_mask, smiles_padded, smiles_mask)
        emb_view_1_mt, emb_view_2_mt, score_mt = self.get_mt_encoder(seqs_mt_padded, seqs_mt_mask, smiles_padded, smiles_mask)

        emb_view_1_mt, emb_view_1_wt = emb_view_1_mt.detach(), emb_view_1_wt.detach()
        return emb_view_1_wt, emb_view_2_wt, emb_view_1_mt, emb_view_2_mt, score_wt, score_mt
