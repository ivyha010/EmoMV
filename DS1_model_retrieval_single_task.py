import torch
from torch import nn
from torch.nn import functional as F
import math


class GELU(nn.Module):
    def forward(self, x):
        temp1 = x.pow(3) # math.pow(x,3)
        temp2 = math.sqrt(2/math.pi) * (x + 0.044715 * x.pow(3))
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
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.fc_embed_1(x)
        x = self.gelu(projected)
        x = self.fc_embed_2(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = x + projected
        x = self.layer_norm(x)
        x = self.gelu(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class video_branch(nn.Module):
    def __init__(
            self,
            video_dim
    ):
        super().__init__()
        # video branch
        self.video_fc = nn.Linear(video_dim, 64)
    def forward(self, x):
        x = self.video_fc(x)
        return x


class audio_branch(nn.Module):
    def __init__(
            self,
            audio_dim
    ):
        super().__init__()
        # audio branch
        self.audio_fc = nn.Linear(audio_dim, 64)
    def forward(self, x):
        x = self.audio_fc(x)
        return x


class embedding_network(nn.Module):
    def __init__(
        self,
        video_dim = 2304,
        audio_dim = 128,
    ):
        super().__init__()
        # video branch
        self.video_br = video_branch(video_dim)

        # audio branch
        self.audio_br= audio_branch(audio_dim)

        # distance learning model
        self.video_projection = ProjectionHead(embedding_dim=64, projection_dim=64)
        self.audio_projection = ProjectionHead(embedding_dim=64, projection_dim=64)

        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def forward(self, video_features, audio_features):
        # video branch
        video_iter = self.video_br(video_features)
        # audio branch
        audio_iter = self.audio_br(audio_features)

        # cross modal distance learning
        normL2_video_embeddings = self.video_projection(video_iter)
        normL2_audio_embeddings = self.audio_projection(audio_iter)

        cosine_sim = self.cosine_sim(normL2_video_embeddings, normL2_audio_embeddings)
        return cosine_sim

