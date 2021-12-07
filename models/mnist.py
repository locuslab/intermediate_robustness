import torch.nn as nn
import torch


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0),-1)


class mnist_classifier(nn.Module):
    def __init__(self, N=32):
        super(mnist_classifier, self).__init__()
        self.net = nn.Sequential(*[
            nn.Conv2d(1, N, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(N, 2 * N, 4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(7 * 7 * 2 * N, 32 * N),
            nn.ReLU(),
            nn.Linear(32 * N, 10)])

    def forward(self, X):
        X = torch.clamp(X, min=0, max=1)
        return self.net(X)
