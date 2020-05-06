import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet34(pretrained=True)

# Must add this line, or there would be BUG, like the output index not right
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("traced_resnet_model.pt")

# with torch.no_grad():
#  output = traced_script_module(torch.ones(1, 3, 224, 224))
#
# print(output[0, :5])
