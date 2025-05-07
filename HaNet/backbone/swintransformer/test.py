import torch
import torch.nn as nn
import timm

model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, in_chans=34)
model.head = nn.Identity()
print(model)

tensor = torch.rand(1,34,224,224)
out = model(tensor)
print(out.shape)