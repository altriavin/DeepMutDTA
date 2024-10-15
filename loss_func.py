import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, wt_emb, mt_emb, label_wt, label_mt):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]
        # print(f'wt_emb.shape: {wt_emb.shape}, mt_emb.shape: {mt_emb.shape}')
        features = torch.cat([wt_emb, mt_emb], dim=0)  # [2bs, feat_dim]
        # labels = torch.cat([torch.ones(labels.shape).cuda(), labels], dim=0).unsqueeze(1)
        # features = mt_emb
        labels = torch.cat([label_wt, label_mt], dim=0).unsqueeze(1)
        # labels = labels.unsqueeze(1)
        # print(f'labels.shape: {labels.shape}')
        # labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        # print(f'label_diffs: {label_diffs}')

        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # print(f'label_diffs: {label_diffs}')

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).cuda()).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).cuda()).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).cuda()).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss
