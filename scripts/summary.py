"""
    This file borrowed from this repo: https://github.com/sksq96/pytorch-summary
    Thanks :)
"""

from collections import OrderedDict

import torch as th
from torch import nn
from torch.autograd import Variable


def summary(input_size, model):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]

            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            summary[m_key]['output_shape'] = list(output.size())
            summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += th.prod(th.LongTensor(list(module.weight.size())))
                if module.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            if hasattr(module, 'bias') and module.bias is not None:
                params += th.prod(th.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if not isinstance(module, nn.Sequential) and \
                not isinstance(module, nn.ModuleList) and \
                not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(th.rand(1, *in_size)) for in_size in input_size]
    else:
        x = Variable(th.rand(1, *input_size))

    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    print('----------------------------------------------------------------')
    line_new = '{:<25}  {:<25} {:<15}'.format('Layer (type)', 'Output Shape', 'Param #')
    print(line_new)
    print('================================================================')
    total_params = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = '{:<25}  {:<25} {:<15}'.format(layer, str(summary[layer]['output_shape']),
                                                  '{0:,}'.format(summary[layer]['nb_params']))
        total_params += summary[layer]['nb_params']
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] == True:
                trainable_params += summary[layer]['nb_params']
        print(line_new)
    print('================================================================')
    print('Total params: {0:,}'.format(total_params))
    print('Trainable params: {0:,}'.format(trainable_params))
    print('Non-trainable params: {0:,}'.format(total_params - trainable_params))
    print('----------------------------------------------------------------')

