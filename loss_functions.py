import torch
import torch.nn
from torch.nn import functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.

    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin


    def forward(self, dist, y):
        mdist = self.margin - dist
        m_mdist = torch.clamp(mdist, min=0.0)
        m_mdist_sq =  torch.pow(m_mdist,2)
        y = y.float()
        loss = (1.0 - y) * m_mdist_sq + y * torch.pow(dist, 2)  # if Y = 1: Match, Y= 0: MISMATCH
        #loss = y * m_mdist_sq + (1.0 - y) * torch.pow(dist, 2)  # if Y = 0: Match, Y= 1: MISMATCH
        loss = torch.sum(loss) / 2.0 / y.size()[0]
        return loss
