import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log.txt space.
        # 5000*512
        pe = torch.zeros(max_len, d_model).float()
        # 标量
        pe.require_grad = False
        # 5000*1
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # 256
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # 5000*256 https://www.cnblogs.com/yanjy-onlyone/p/11553084.html
        # import torch
        # a = torch.zeros(2, 4).float()
        # b = torch.rand(2, 2)
        # a[:, 0::2] = b
        # print(a.shape)
        # print(b.shape)
        # print(a)
        # torch.Size([2, 4])
        # torch.Size([2, 2])
        # print(a.unsqueeze(0))
        # print(a.unsqueeze(1))
        # tensor([[0.6014, 0.0000, 0.2836, 0.0000],
        #         [0.8686, 0.0000, 0.5273, 0.0000]])
        # tensor([[[0.3927, 0.0000, 0.6801, 0.0000],
        #          [0.9745, 0.0000, 0.0928, 0.0000]]])
        # tensor([[[0.3927, 0.0000, 0.6801, 0.0000]],
        #
        #         [[0.9745, 0.0000, 0.0928, 0.0000]]])
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # squeeze/unsqueeze挤压和增加维度的操作
        pe = pe.unsqueeze(0)
        # 使用别名
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 32*96*7(x) -> 96(x.size(1) seq_len) -> 1*96*512(output)
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 一维卷积 c_in=7(input_dim), d_model=512(seq_len)
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 使用hekaiming的初始化方法
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # 32*96*7 -> 96*7*32(维度交换, seq_len, input_channel ,batch_size) -> 32*512*96(一维卷积降采样：batch_size, output_channel, seq_len) -> 32*96*512(转置)
        # CONV1D NCL->NCL Lout=func(Lin) https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        # batch_size不变 input_channel和output_channel自定义，seq_len增加
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        # 经过con1d之后 32*96*512
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    时间编码
    """

    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        # 时间特征编码
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        # 查询时间特征的字典
        d_inp = freq_map[freq]
        # 返回线性层
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        # 32*96*4 -> 32, 96, 512
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        # Token 词语/字编码 使用卷积编码 c_in = 7 d_model = 512
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # Positional 位置编码
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # temporal 时间编码：两种方式 TimeFeatureEmbedding 或者 TemporalEmbedding
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) \
            if embed_type != 'timeF' \
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # 三种编码加法 torch.Size([32, 96, 512]) + torch.Size([1, 96, 512]) + torch.Size([32, 96, 512])
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
