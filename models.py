"""
Model architectures for audio classification.
Contains CNN model and CNN-Transformer model.
"""

import torch
import torch.nn as nn


class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=26, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(26),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv1d(in_channels=26, out_channels=52, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(52),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(0.25),

            nn.Conv1d(in_channels=52, out_channels=104, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(104),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv1d(in_channels=104, out_channels=26, kernel_size=2, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(26),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv1d(in_channels=26, out_channels=20, kernel_size=2, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Flatten()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(880, 512),
            nn.Sigmoid(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class CnnFeatureExtractor(nn.Module):
    """CNN feature extractor without classification head for transformer input"""
    def __init__(self):
        super(CnnFeatureExtractor, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=26, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(26),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv1d(in_channels=26, out_channels=52, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(52),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(0.25),

            nn.Conv1d(in_channels=52, out_channels=104, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(104),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv1d(in_channels=104, out_channels=26, kernel_size=2, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(26),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv1d(in_channels=26, out_channels=20, kernel_size=2, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Flatten())

    def forward(self, x):
        x = self.conv_layers(x)
        return x


class PositionEncoder(nn.Module):
    def __init__(self, max_len=880):
        super(PositionEncoder, self).__init__()
        self.max_len = max_len
        self.encoding = None

    def generate_encoding(self, pos):
        position = torch.arange(0, pos).unsqueeze(1)
        p = position / (10000 ** (torch.arange(0, self.max_len).view(-1, 2)[:, 0].repeat_interleave(2) / self.max_len))
        encoding = torch.zeros(pos, self.max_len)
        encoding[::2, :] = (encoding[::2, :] + torch.sin(p[::2, :]))
        encoding[1::2, :] = (encoding[1::2, :] + torch.cos(p[1::2, :]))
        return encoding

    def forward(self, x):
        pos = x.shape[0]
        if self.encoding is None or self.encoding.size(0) != pos:
            self.encoding = self.generate_encoding(pos)
        self.encoding = self.encoding.to(x.device)

        return x + self.encoding[:, :x.size(1)].detach()


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        Q = self.WQ(query).view(query.size(0), -1, self.num_heads, self.head_dim)
        K = self.WK(key).view(key.size(0), -1, self.num_heads, self.head_dim)
        V = self.WV(value).view(value.size(0), -1, self.num_heads, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", Q, K)
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        x = torch.einsum("nhql,nlhd->nqhd", attention, V).contiguous()
        x = x.view(x.size(0), -1, self.d_model)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.attention_blocks = nn.ModuleList([
            SelfAttention(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for attention_block in self.attention_blocks:
            residual = x.clone()
            x = self.norm1(x)
            x = attention_block(x, x, x)
            x = self.dropout(x)
            x = x + residual
            x = self.norm2(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


class CnnTransformerModel(nn.Module):
    def __init__(self, input_channels, output_channels, d_model, num_heads, num_layers, num_classes, dropout):
        super(CnnTransformerModel, self).__init__()
        self.base_network = CnnFeatureExtractor()
        self.positional_encoding = PositionEncoder()
        self.transformer_encoder = TransformerEncoder(d_model, num_heads, num_layers, dropout)
        self.mlp_head = MLPHead(d_model, num_classes)

    def forward(self, x):
        x = self.base_network(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        cls_token = x[:, 0, :]
        output = self.mlp_head(cls_token)
        return output
