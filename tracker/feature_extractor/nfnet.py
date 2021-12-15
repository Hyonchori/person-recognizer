import timm
import torch

class Nfnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = timm.create_model("efficientnetv2_s", pretrained=True)
        self.base.reset_classifier(0)

    def forward(self, x):
        x = self.base(x)
        return x


if __name__ == "__main__":
    model = Nfnet()
    sample = torch.randn(1, 3, 224, 224)
    pred = model(sample)
    print(pred.shape)