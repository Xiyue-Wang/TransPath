import torch
from ctran import ctranspath



net = ctranspath()
td = torch.load(r'./ctranspath.pth')
net.load_state_dict(td['model'], strict=False)

linear_keyword = 'head'

for name, param in net.named_parameters():
    if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
        param.requires_grad = False
# init the fc layer
getattr(net, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
getattr(net, linear_keyword).bias.data.zero_()

parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
assert len(parameters) == 2  # weight, bias

####train your task
