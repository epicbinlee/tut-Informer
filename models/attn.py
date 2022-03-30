from math import sqrt

import numpy as np
import torch
import torch.nn as nn

from utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        # queries:torch.Size([32, 72, 8, 64]) value:torch.Size([32, 48, 8, 64])
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        # 全文最难理解：爱因斯坦矩阵乘法标记 torch.Size([32, 8, 72, 48]) = torch.Size([32, 72, 8, 64]) * torch.Size([32, 48, 8, 64])
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # 使用mask
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        # dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # 矩阵乘法 torch.Size([32, 72, 8, 64]) = torch.Size([32, 8, 72, 48]) * torch.Size([32, 48, 8, 64])
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # 输出
        if self.output_attention:
            # contiguous 改变了数据在内存中形态，保持连续的内存
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D] torch.Size([32, 8, 96, 64]) batch_size, multi_head, seq_length, head_dim
        B, H, L_K, E = K.shape
        # Q [B, H, L, D] torch.Size([32, 8, 96, 64]) batch_size, multi_head, seq_length, head_dim
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K, K torch.Size([32, 8, 96, 64]), K_expand torch.Size([32, 8, 96, 96, 64])
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # torch.Size([96, 25])
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        # torch.Size([32, 8, 96, 25, 64])
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        # torch.Size([32, 8, 96, 25]) = torch.Size([32, 8, 96, 1, 64]) * torch.Size([32, 8, 96, 64, 25])
        # x = torch.ones(32, 8, 96, 1, 64)
        # y = torch.ones(32, 8, 96, 64, 25)
        # z = torch.matmul(x, y)
        # print(z.shape)
        # assert z.shape == (32, 8, 96, 1, 25)
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        # find the Top_k query with sparisty measurement，torch.Size([32, 8, 96]) = torch.Size([32, 8, 96]) - torch.Size([32, 8, 96])
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        # torch.Size([32, 8, 25])
        M_top = M.topk(n_top, sorted=False)[1]
        # use the reduced Q to calculate Q_K torch.Size([32, 8, 25, 64])
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]  # factor*ln(L_q)
        # torch.Size([32, 8, 25, 96]) = torch.Size([32, 8, 25, 64]) * torch.Size([32, 8, 64, 96])
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
        # scores_top, index = torch.Size([32, 8, 25, 96]) torch.Size([32, 8, 25])
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        # torch.Size([32, 96, 8, 64])
        B, L_Q, H, D = queries.shape
        # torch.Size([32, 96, 8, 64])
        _, L_K, _, _ = keys.shape
        # torch.Size([32, 8, 96, 64])
        queries = queries.transpose(2, 1)
        # torch.Size([32, 8, 96, 64])
        keys = keys.transpose(2, 1)
        # torch.Size([32, 8, 96, 64])
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        # 计算评分最高的下标， 评分计算方法参考论文，简单来说就是就KQ和均匀分布的差异性
        # scores_top, index = torch.Size([32, 8, 25, 96]) torch.Size([32, 8, 25])
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top *= scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        # torch.Size([32, 96, 512])
        B, L, _ = queries.shape
        # torch.Size([32, 96, 512])
        _, S, _ = keys.shape
        # 8
        H = self.n_heads

        # 输入数据 三个线性变换得到 QKV矩阵
        # torch.Size([32, 96, 8, 64])
        queries = self.query_projection(queries).view(B, L, H, -1)
        # torch.Size([32, 96, 8, 64])
        keys = self.key_projection(keys).view(B, S, H, -1)
        # torch.Size([32, 96, 8, 64])
        values = self.value_projection(values).view(B, S, H, -1)

        # 根据不同的Attention机制做计算
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        # Attention的格式
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
