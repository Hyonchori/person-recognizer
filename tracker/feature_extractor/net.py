import torch
import torch.nn as nn

from net_blocks import initialize_weights, Conv, Residual
from utils import model_info


class EfficientNet(torch.nn.Module):
    """
    EfficientNetV2: Smaller Models and Faster Training
    https://arxiv.org/pdf/2104.00298.pdf
    """

    def __init__(self, args) -> None:
        super().__init__()
        gate_fn = [True, False]
        filters = [24, 48, 64, 128, 160, 272, 1792]

        feature = [Conv(args, 3, filters[0], torch.nn.SiLU(), 3, 2)]
        if args:
            filters[5] = 256
            filters[6] = 1280
        for i in range(2):
            if i == 0:
                feature.append(
                    Residual(args, filters[0], filters[0], 1, 1, gate_fn[0]))
            else:
                feature.append(
                    Residual(args, filters[0], filters[0], 1, 1, gate_fn[0]))

        for i in range(4):
            if i == 0:
                feature.append(
                    Residual(args, filters[0], filters[1], 2, 4, gate_fn[0]))
            else:
                feature.append(
                    Residual(args, filters[1], filters[1], 1, 4, gate_fn[0]))

        for i in range(4):
            if i == 0:
                feature.append(
                    Residual(args, filters[1], filters[2], 2, 4, gate_fn[0]))
            else:
                feature.append(
                    Residual(args, filters[2], filters[2], 1, 4, gate_fn[0]))

        for i in range(6):
            if i == 0:
                feature.append(
                    Residual(args, filters[2], filters[3], 2, 4, gate_fn[1]))
            else:
                feature.append(
                    Residual(args, filters[3], filters[3], 1, 4, gate_fn[1]))

        for i in range(9):
            if i == 0:
                feature.append(
                    Residual(args, filters[3], filters[4], 1, 6, gate_fn[1]))
            else:
                feature.append(
                    Residual(args, filters[4], filters[4], 1, 6, gate_fn[1]))

        for i in range(15):
            if i == 0:
                feature.append(
                    Residual(args, filters[4], filters[5], 2, 6, gate_fn[1]))
            else:
                feature.append(
                    Residual(args, filters[5], filters[5], 1, 6, gate_fn[1]))
        feature.append(Conv(args, filters[5], filters[6], torch.nn.SiLU()))

        self.feature = torch.nn.Sequential(*feature)

        initialize_weights(self)

    def forward(self, x):
        x = self.feature(x)
        return x

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model'].state_dict()

        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, neck="bnneck", neck_feat="after", args=True, info=False):
        super().__init__()
        self.base = EfficientNet(args)
        self.in_planes = 1280
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == "no":
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
        elif self.neck == "bnneck":
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        if info:
            model_info(self.base, verbose=True, input_shape=(3, 224, 224), batch_size=32)

    def forward(self, x):
        global_feat = self.gap(self.base(x))
        global_feat = global_feat.view(global_feat.shape[0], -1)
        feat = self.bottleneck(global_feat) if self.neck == "bnneck" else global_feat
        cls_score = self.classifier(feat)
        return cls_score, global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path).state_dict()
        for i in param_dict:
            if "classifier" in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


if __name__ == "__main__":
    model = EfficientNetClassifier(num_classes=255, info=True)
    sample = torch.randn(12, 3, 256, 256)
    score, feat = model(sample)
    print(score.shape, feat.shape)