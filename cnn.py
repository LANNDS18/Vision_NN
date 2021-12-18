'''
    Step 1 and Step 2 is done in this file.
    calss CNN is builded as step 1 requires.
    function createLossAndOptimizer is builded as step 2 requires.
'''

from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batch_norm_1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.fc = nn.Linear(64 * 30 * 30, 5)

    def forward(self, x):
        """
        Forward pass,
        x shape is (batch_size, 3, 250, 250)
        (color channel first)
        in the comments, we omit the batch_size in the shape
        """
        # shape : 3x250x250 -> 64x125x125 (Recall: (N-F+2P)/S+1) = (250-7+2*3)/2+1 = 125
        x = self.conv1(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)
        # 64x125x125 -> 64x62x62 (Recall: (N -F+2P)/S+1) = (125-3+2*0)/2+1 = 62
        x = self.pool1(x)

        # shape : 64x62x62 -> 64x62x62 ((62-3+2*1)/1+1) = 62
        x = self.conv2(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)
        # 64x62x62 -> 64x30x30 (Recall: (N-F+2P)/S+1) = (62-3+2*0)/2+1 = 30
        x = self.pool2(x)

        # 64x30x30 -> 57600
        x = x.view(-1, 64 * 30 * 30)
        # 57600 -> 5
        x = self.fc(x)
        # The softmax non-linearity is applied later (cf createLossAndOptimizer() fn)
        return x


def createLossAndOptimizer(net, learning_rate=0.001):
    # it combines softmax with negative log likelihood loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    return criterion, optimizer
