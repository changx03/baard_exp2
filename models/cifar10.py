import torch
import torch.nn as nn
import torchvision as tv


class Resnet(nn.Module):
    def __init__(self, n_labels=10, use_prob=True):
        super(Resnet, self).__init__()
        self.n_labels = n_labels
        self.use_prob = use_prob

        # Load pre-trained resnet model
        resnet = tv.models.resnet18(pretrained=True, progress=False)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        n_in_features = resnet.fc.in_features
        self.fc = nn.Linear(n_in_features, self.n_labels)
        self.softmax = nn.Softmax(dim=1)

    def before_softmax(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.before_softmax(x)
        if self.use_prob:
            x = self.softmax(x)
        return x


class Vgg(nn.Module):
    def __init__(self, n_labels=10, use_prob=True):
        super(Vgg, self).__init__()
        self.n_labels = n_labels
        self.use_prob = use_prob

        # Load pre-trained vgg model
        vgg = tv.models.vgg11(pretrained=True, progress=False)

        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
        self.classifier[-1] = nn.Linear(4096, self.n_labels)
        self.softmax = nn.Softmax(dim=1)

    def before_softmax(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.before_softmax(x)
        if self.use_prob:
            x = self.softmax(x)
        return x


def test(model):
    model.to('cpu')
    X = torch.randn(5, 3, 32, 32, dtype=torch.float32)
    y = model(X)
    print(y.size())


if __name__ == '__main__':
    test(Resnet())
    test(Vgg())
