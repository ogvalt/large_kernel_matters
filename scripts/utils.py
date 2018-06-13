import warnings

import torch as th
from torch.nn.functional import pad
from torch.nn import Module, Conv2d


class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same',
                 stride=1, dilation=1, groups=1):
        super(Conv2D, self).__init__()

        assert type(kernel_size) in [int, tuple], "Allowed kernel type [int or tuple], not {}".format(type(kernel_size))
        assert padding in ['valid', 'same'], "Allowed padding type {}, not [{}]".format(['valid', 'same'], padding)
        if padding == 'same' and stride > 1:
            warnings.warn("In case of 'padding == 'same' and stride > 1' input image or feature map will scaled " +
                          "{} times in size due to padding".format(stride))

        self.kernel_size = kernel_size

        if isinstance(kernel_size, tuple):
            self.h_kernel = kernel_size[0]
            self.w_kernel = kernel_size[1]
        else:
            self.h_kernel = kernel_size
            self.w_kernel = kernel_size

        self.padding = padding
        self.stride = stride

        self.dilation = dilation
        self.groups = groups

        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=self.stride, dilation=self.dilation, groups=self.groups)

    def forward(self, x):

        if self.padding == 'same':

            height, width = x.shape[2:]

            h_pad_need = max(0, (height - 1) * self.stride + self.h_kernel - height)
            w_pad_need = max(0, (width - 1) * self.stride + self.w_kernel - width)

            pad_left = w_pad_need // 2
            pad_right = w_pad_need - pad_left
            pad_top = h_pad_need // 2
            pad_bottom = h_pad_need - pad_top

            padding = (pad_left, pad_right, pad_top, pad_bottom)

            x = pad(x, padding, 'constant', 0)

        x = self.conv(x)

        return x


def test_Conv2D():

    from torch.autograd import Variable

    for input_size in [(1, 100 + step, 100 + step) for step in range(10, 300, 10)]:

        for kernel_size in [(k, k) for k in range(3, 17, 2)]:

            for stride in range(1, 4, 1):

                print("------------Start---------------")
                print("Input size", input_size)
                print("Kernel_size", kernel_size)
                print("Stride", stride)
                print()
                c = Conv2D(in_channels=1, out_channels=1, kernel_size=(5, 5), padding='same', stride=stride)
                if isinstance(input_size[0], (list, tuple)):
                    x = [Variable(th.rand(1, *in_size)) for in_size in input_size]
                else:
                    x = Variable(th.rand(1, *input_size))
                c(x)
                print("------------End-----------------")


if __name__ == '__main__':
    test_Conv2D()