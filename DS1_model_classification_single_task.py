import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from numpy import dot
from numpy.linalg import norm
from torch.autograd import Variable
import math
from compact_bilinear_pooling import CompactBilinearPooling


class GELU(nn.Module):
    def forward(self, x):
        gelu = 0.5 * x * (1 + F.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x.pow(3))))
        return gelu

class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim,
            dropout=0.5
    ):
        super().__init__()
        self.fc_embed_1 = nn.Linear(embedding_dim, projection_dim)
        self.fc_embed_2 = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = GELU()

    def forward(self, x):
        projected = self.fc_embed_1(x)
        x = self.gelu(projected)
        x = self.fc_embed_2(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = x + projected
        x = self.gelu(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class embedding_network(nn.Module):
    def __init__(
        self,
        video_dim = 2304,
        audio_dim = 128,
        cbp_dim=32

    ):
        super().__init__()
        # video branch
        self.video_fc = nn.Linear(video_dim, 64)

        # audio branch
        self.audio_fc= nn.Linear(audio_dim, 64)

        self.gelu = GELU()

        # distance learning model
        self.video_projection = ProjectionHead(embedding_dim=64, projection_dim=64)
        self.audio_projection = ProjectionHead(embedding_dim=64, projection_dim=64)
        self.cbp = CompactBilinearPooling(64, 64, cbp_dim)
        self.cbp_fc1 = nn.Linear(cbp_dim, 32)
        self.cbp_fc2 = nn.Linear(32, 2)

    def forward(self, video_features, audio_features):
        # video branch
        video_iter = self.video_fc(video_features)
        # audio branch
        audio_iter = self.audio_fc(audio_features)

        normL2_video_embeddings = self.video_projection(video_iter)
        normL2_audio_embeddings = self.audio_projection(audio_iter)
        cbp = self.cbp(normL2_video_embeddings, normL2_audio_embeddings)
        out_match = self.cbp_fc2(self.gelu(self.cbp_fc1(cbp)))

        return out_match.squeeze(1)

