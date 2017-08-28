import torch

def info(prefix, var):
    print('-------{}----------'.format(prefix))
    if isinstance(var, torch.autograd.variable.Variable):
        print('Variable:')
        print('size: ', var.data.size())
        print('data type: ', type(var.data))
    elif isinstance(var, torch.FloatTensor) or isinstance(var, torch.cuda.FloatTensor):
        print('Tensor:')
        print('size: ', var.size())
        print('type: ', type(var))
    else:
        print(type(var))