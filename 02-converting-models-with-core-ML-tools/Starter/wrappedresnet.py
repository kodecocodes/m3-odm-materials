class WrappedResnet(torch.nn.Module):
  def __init__(self):
    super(WrappedResnet, self).__init__()
    self.model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True).eval()

  def forward(self, x):
    res = self.model(x)
    x = res["out"]
    return x
  