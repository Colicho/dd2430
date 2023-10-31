import torch.nn as nn

class SiameseNetworkSimple(nn.Module):
    def __init__(self):
        super(SiameseNetworkSimple, self).__init__()
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
        super(SiameseNetworkComplex, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Linear(21, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.batch_norm = nn.BatchNorm1d(256)
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.9),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.batch_norm(x)
        output = self.fc_layers(x)
        return output
    