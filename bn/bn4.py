import numpy as np
import mindspore

from mindspore import context
from mindspore.common.tensor import Tensor
import mindspore.ops.operations as P
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.nn import Cell
from mindspore.nn import BatchNorm2d
import torch
import torchvision

class Batchnorm_Net(Cell):
    def __init__(self, c, weight, bias, moving_mean, moving_var_init, use_batch_statistics=None):
        super(Batchnorm_Net, self).__init__()
        self.bn = BatchNorm2d(c, eps=0.00001, momentum=0.1, beta_init=bias, gamma_init=weight,
                              moving_mean_init=moving_mean, moving_var_init=moving_var_init,
                              use_batch_statistics=use_batch_statistics)

    def construct(self, input_data):
        x = self.bn(input_data)
        return x


class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, sens):
        gout = self.grad(self.network)(input_data, sens)
        return gout

class TorchNet(torch.nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()
        self.bn = torch.nn.BatchNorm2d(3, affine=False)

    def forward(self, x):
        return self.bn(x)

# Prepare data
#np.random.seed(1)
x_np = np.random.randn(1, 3, 2, 2).astype(np.float32)
input_grad_np = np.random.randn(1, 3, 2, 2).astype(np.float32)

# torch
torch_input = torch.from_numpy(x_np)
torch_input.requires_grad=True
torch_net = TorchNet()
torch_out = torch_net(torch_input)
torch_out_np = torch_out.detach().numpy()
torch_out.backward(torch.from_numpy(input_grad_np))
torch_out_grad_np = torch_input.grad.data.clone().detach().cpu().numpy()

#  mindspore 
ms_input = mindspore.Tensor(x_np)
weight = Tensor(np.ones(3).astype(np.float32))
bias = Tensor(np.zeros(3).astype(np.float32))
moving_mean = Tensor(np.zeros(3).astype(np.float32))
moving_var_init = Tensor(np.ones(3).astype(np.float32))

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
ms_net = Batchnorm_Net(3, weight, bias, moving_mean, moving_var_init)
ms_net.set_train()
ms_out = ms_net(ms_input)
ms_grad = Grad(ms_net)
ms_out_grad_np = ms_grad(Tensor(ms_input), Tensor(input_grad_np))

# Compare
N = 5
print("Print the first N = ", N)
print("torch  out ", torch_out_np.reshape(-1)[:N])
print("minds  out ", ms_out.asnumpy().reshape(-1)[:N])
print("torch grad ", torch_out_grad_np.reshape(-1)[:N])
print("minds grad ", ms_out_grad_np[0].asnumpy().reshape(-1)[:N])

assert np.allclose(ms_out.asnumpy(), torch_out_np)
assert np.allclose(ms_out_grad_np[0].asnumpy(), torch_out_grad_np)
