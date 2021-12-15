import torchvision
import torch
import os
from net.models.modeling import VisionTransformer, CONFIGS
num_classes =1000
arg = CONFIGS['R50-ViT-B_16']
model = VisionTransformer(arg, 256, zero_head=True, num_classes=num_classes)
# print(model)
state_dict = torch.load(r'./checkpoint.pth')

for k in list(state_dict.keys()):
    # retain only base_encoder up to before the embedding layer
    if k.startswith('module.online_encoder.net') and not k.startswith('module.online_encoder.net.head'):
        # remove prefix
        state_dict[k[len("module.online_encoder.net."):]] = state_dict[k]
    # delete renamed or unused k
    del state_dict[k]
model.load_state_dict(state_dict, strict=False)

torch.save(state_dict, r'/R50-ViT-B_16.pth')
