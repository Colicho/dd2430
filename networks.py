import torch.nn as nn

class SiameseNetworkSimple(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(21, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        output = self.fc(x)
        return output
    
class SiameseNetworkComplex(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Linear(21, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )
        self.batch_norm = nn.BatchNorm1d(256)
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.9),
            nn.Linear(128, 64)
        )

    def forward_once(self, x):
        x = self.cnn_layers(x)
        x = self.batch_norm(x)
        x = self.fc_layers(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2