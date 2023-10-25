import torch
import torch.nn as nn


class ContrastiveLossSimple(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLossSimple, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.PairwiseDistance(p=2)
        distance = euclidean_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss_contrastive
    

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
        # return torch.div(input, torch.add(input, 1))
        return input