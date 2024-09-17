mkdir lesson-two
cd lesson-two

python

pt_model = \
  torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', \
  pretrained=True)

model = pt_model.eval()

im = torch.zeros(1, 3, 640, 640)

## Also available in starter project as wrappedresnet.py
class WrappedResnet(torch.nn.Module):
  def __init__(self):
    super(WrappedResnet, self).__init__()
    self.model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True).eval()

  def forward(self, x):
    res = self.model(x)
    x = res["out"]
    return x

traceable_model = WrappedResnet().eval()
trace = torch.jit.trace(torch_model, im)

mlmodel = ct.convert(
  trace,
  inputs=[ct.TensorType(name="input", shape=im.shape)]
)

mlmodel.save("newmodel.mlpackage")
