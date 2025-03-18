import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from fast_transformer import FastformerEncoder, AttentionPooling

class ExactTopKAttention(nn.Module):
    """Implement the oracle top-k softmax attention.

    Arguments
    ---------
        top-k: The top k keys to attend to  (default: 32)
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, hidden_size, topk=32, n_head=4, attention_dropout=0.1):
        super(ExactTopKAttention, self).__init__()
        self.topk = topk
        self.f_q = nn.Linear(hidden_size, hidden_size)
        self.f_k = nn.Linear(hidden_size, hidden_size)
        self.f_v = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size
        self.n_heads = n_head
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        
        queries = self.f_q(queries)
        keys = self.f_k(keys)
        values = self.f_v(values)

        queries = queries.view(queries.shape[0], queries.shape[1], self.n_heads, self.hidden_size // self.n_heads)
        keys = keys.view(keys.shape[0], keys.shape[1], self.n_heads, self.hidden_size // self.n_heads)
        values = values.view(values.shape[0], values.shape[1], self.n_heads, self.hidden_size // self.n_heads)

        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = 1./math.sqrt(E)

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhe,nshe->nhls", queries, keys)
        topk = min(self.topk, S)
        
        # QK = QK + torch.log(query_lengths.bool().float())[:, None, None]
        # print(f'QK.sahpe; {QK.shape}')

        topk_values, topk_idx = torch.topk(QK, topk, sorted=False, dim=-1)
        mask = QK.new_ones(QK.shape) *  float("-inf") 
        mask[
            torch.arange(N, device=QK.device).view(N, 1, 1, 1),
            torch.arange(H, device=QK.device).view(1, H, 1, 1),
            torch.arange(L, device=QK.device).view(1, 1, L, 1),
            topk_idx,
        ] = 0.

        QK = QK + mask 

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        # print(f'A.shape: {A.shape}; values.shape: {values.shape}')
        V = torch.einsum("nhls,nshd->nlhd", A, values)
        V = V.reshape(V.shape[0], V.shape[1], -1)
        # Make sure that what we return is contiguous
        return V.contiguous(), A


class ProteinEncoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, hidden_size, n_layers, kernel_size, dropout):
        super().__init__()

        assert kernel_size % 2 == 1

        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.seq_emb = nn.Embedding(vocab_size, hidden_size).cuda()
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).cuda()
        self.convs = nn.ModuleList([nn.Conv1d(hidden_size, 2 * hidden_size, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, protein):
        # pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).cuda()
        # protein = protein + self.pos_embedding(pos)
        # print(f'protein: {protein}')
        protein = self.seq_emb(protein)
        conv_input = protein.permute(0, 2, 1).cuda()
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved
        conved = conved.permute(0,2,1)
        return conved


class DrugEncoder(nn.Module):
    def __init__(self, max_drug_len, hidden_size):
        super().__init__()

        self.max_drug_len = max_drug_len
        # print(f'self.max_drug_len: {self.max_drug_len}')
        self.model = AutoModel.from_pretrained("/root_path/MoLFormer", trust_remote_code=True).cuda()
        self.fc = nn.Linear(768, hidden_size).cuda()

    def forward(self, smiles_padded, smiles_mask):
        drug_emb = self.model(smiles_padded, smiles_mask)
        # print(f'drug_emb: {drug_emb}')
        drug_emb = self.fc(drug_emb.last_hidden_state)
        # return 
        return drug_emb


class FusionEmbConcat(nn.Module):
    def __init__(self):
        super(FusionEmbConcat, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).cuda()

    def forward(self, drug_emb, seqs_emb):
        drug_emb, seqs_emb = torch.mean(drug_emb, dim=1), torch.mean(seqs_emb, dim=1)
        emb = torch.cat((drug_emb, seqs_emb), 1)
        emb = F.normalize(emb, p=2, dim=1)
        # print(f'emb: {emb}')
        # print(f'emb.shape: {emb.shape}')
        output = self.fc(emb)
        return output
    
    
class TransformerModel(nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()

        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads 
        self.max_seq_len = args.max_seq_len
        self.max_drug_len = args.max_drug_len
        self.topk = args.topk
        self.task_type = args.task_type

        self.protein_encode = FastformerEncoder(args=args).cuda()
        self.drug_encode = DrugEncoder(self.max_drug_len, self.hidden_size)
        
        self.fusion_layer = ExactTopKAttention(self.hidden_size, topk=self.topk, n_head=self.num_heads, attention_dropout=0.1).cuda()
        self.attention_pool = AttentionPooling(args)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1)
        ).cuda()

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, smiles_padded, seqs_padded, smiles_mask, seqs_mask):
        seqs_emb = self.protein_encode(seqs_padded, seqs_mask)
        smiles_emb = self.drug_encode(smiles_padded, smiles_mask)
        seqs_emb, smiles_emb = F.normalize(seqs_emb, p=2, dim=1), F.normalize(smiles_emb, p=2, dim=1)

        smiles_output, smiles_attention = self.fusion_layer(seqs_emb, smiles_emb, smiles_emb)
        smiles_output, smiles_attn_pool = self.attention_pool(smiles_output)
        seqs_output, seqs_attention = self.fusion_layer(smiles_emb, seqs_emb, seqs_emb)
        seqs_output, seqs_attn_pool = self.attention_pool(seqs_output)

        emb = torch.cat((smiles_output, seqs_output), 1)
        emb = F.normalize(emb, p=2, dim=1)
        output = self.fc(emb)
        if self.task_type == "clf":
            output = torch.sigmoid(output)
        
        return emb, output.squeeze(1), smiles_attn_pool, seqs_attn_pool
        