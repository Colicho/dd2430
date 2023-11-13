import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.l2norm = nn.PairwiseDistance(p=2)

    def forward(self, output_1, output_2, ref_1, ref_2):
        embeddingDist = torch.pow(self.l2norm(output_1, output_2), 2)
        refDist = torch.pow(self.l2norm(self.referenceTransform(ref_1), self.referenceTransform(ref_2)), 2)

        lossTensor = torch.pow(embeddingDist - refDist, 2)
        return torch.mean(lossTensor)

    def referenceTransform(self, input):
        return input