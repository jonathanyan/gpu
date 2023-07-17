import numpy as np
import mindspore

from mindspore import context
import mindspore.ops.operations as P
import mindspore.nn as nn
from mindspore.ops import composite as C
import torch
import torchvision


np.random.seed(1)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)
# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)


class MsGrad(nn.Cell):
    """Grad."""

    def __init__(self, network):
        """
        Implementation of computing gradient in mindspore
        Args:
            network (Cell): Mindspore network.
        """
        super(MsGrad, self).__init__()
        # self.grad = C.GradOperation(name="get_all", get_all=True,
        #                             sens_param=True)
        self.grad = C.GradOperation(get_all=True,
                                    sens_param=True)
        self.network = network

    def construct(self, *input_data):
        """
        Construct.

        Args:
            input_data: Input data.

        """
        gout = self.grad(self.network)(*input_data)

        return gout


class MsNet(nn.Cell):
    """Net for MindSpore."""

    def __init__(self, **kwargs):
        """
        Python special method __init__ for Net.

        """
        super(MsNet, self).__init__()
        self.op = P.ROIAlign(**kwargs)

    def construct(self, features, rois):
        """
        Construct.

        Args:
            features: Features.
            rois: Rois.
        """
        return self.op(features, rois)


class TorchNet(torch.nn.Module):
    """net_torch."""

    def __init__(self, **kwargs):
        """
        Python special method __init__ for TorchNet.

        """
        super(TorchNet, self).__init__()
        self.kwargs = kwargs
        self.output_size = kwargs.get('output_size')
        self.sampling_ratio = kwargs.get('sampling_ratio')
        self.spatial_scale = kwargs.get('spatial_scale')
        self.aligned = kwargs.get('aligned')

    def forward(self, input_x, boxes):
        """
        Forward.

        Args:
            input_x: Input_x.
            boxes: Boxes.

        """
        return torchvision.ops.roi_align(input_x, boxes, spatial_scale=self.spatial_scale,
                                         output_size=self.output_size, sampling_ratio=self.sampling_ratio,
                                         aligned=self.aligned)


def test_float32():
    np.random.seed(5)
    data_type = np.float32
    device = torch.device('cuda:0')
    print('data_type', data_type)


    pooled_height = 7
    pooled_width = 7
    spatial_scale = 1.0/16.0
    sample_num = 2
    roi_end_mode = 1
    ms_kwargs = dict(pooled_width=pooled_width, pooled_height=pooled_height, spatial_scale=spatial_scale,
                     roi_end_mode=roi_end_mode, sample_num=sample_num)
    torch_kwargs = dict(output_size=(pooled_height, pooled_width), spatial_scale=spatial_scale,
                        sampling_ratio=2, aligned=True)

    #input_shape = (1, 1, 2, 2)
    #input_tensor = np.random.rand(*input_shape).astype(data_type)
    #rois = np.random.randn(2, 5).astype(data_type)

    input_tensor = np.fromfile('Default--network--_backbone--fpn_ncek--fpn_convs_list--2--BiasAdd-op303_output_0_shape_2_256_48_80_kNumberTypeFloat32_DefaultFormat.bin', np.float32)
    rois = np.fromfile('Default--backbone--Concat-op2494_output_0_shape_1280_5_kNumberTypeFloat32_DefaultFormat.bin', np.float32)
    input_tensor = input_tensor.reshape(2,256,48,80)
    rois = rois.reshape(1280,5)


    ms_input = mindspore.Tensor(input_tensor.copy())
    ms_rois = mindspore.Tensor(rois.copy())

    ms_net = MsNet(**ms_kwargs)
    ms_infer_out = ms_net(ms_input, ms_rois)
    ms_infer_out_np = ms_infer_out.asnumpy()

    #d_out = np.load('ms_infer_out_np_d.npy')
    #print('d compare infer')
    #print(np.allclose(ms_infer_out_np, d_out, atol=0.001, rtol=0.001))

    dump_out = np.fromfile('Default--network--_backbone--roi_align--roi_layers--2--ROIAlign-op2528_output_0_shape_1280_256_7_7_kNumberTypeFloat32_DefaultFormat.bin', np.float32)
    dump_out = dump_out.reshape(1280,256,7,7)


    print('dump compare infer')
    print(np.allclose(ms_infer_out_np, dump_out, atol=0.001, rtol=0.001))


    torch_input = torch.tensor(input_tensor.copy(), requires_grad=True, device=device)
    torch_boxes = torch.tensor(rois.copy(), requires_grad=True, device=device)
    torch_net = TorchNet(**torch_kwargs).to(device=device)
    torch_out = torch_net(torch_input, torch_boxes)
    torch_infer_out_np = torch_out.clone().detach().cpu().numpy()

    # print("x ", torch_input)
    # print("boxes ", torch_boxes)

    assert ms_infer_out_np.shape == torch_infer_out_np.shape
    print('torch compare infer')
    # print('ms_infer_out', ms_infer_out_np)
    # print('\n\ntorch_torch_out', torch_infer_out_np)
    print(np.allclose(ms_infer_out_np, torch_infer_out_np, atol=0.001, rtol=0.001))

    #input_grad = np.random.randn(*ms_infer_out_np.shape).astype(data_type)

    input_grad = np.fromfile('Gradients--Default--network--_backbone--bbox_assigner_sampler_for_rcnn--gradSelect--Select-op4847_output_0_shape_1280_256_7_7_kNumberTypeFloat32_DefaultFormat.bin', np.float32)
    input_grad = input_grad.reshape(1280,256,7,7)

    ms_grad_net = MsGrad(ms_net)
    ms_grad_net.set_train()
    ms_grad_out = ms_grad_net(ms_input, ms_rois, mindspore.Tensor(input_grad))

    if isinstance(ms_grad_out, tuple):
        ms_grad_out = ms_grad_out[0]
    ms_grad_out_np = ms_grad_out.asnumpy()

    #d_grad = np.load('ms_grad_out_np_d.npy')
    #print('d compare train')
    #print(np.allclose(ms_grad_out_np, d_grad, atol=0.001, rtol=0.001))


    dump_grad = np.fromfile('Gradients--Default--network--_backbone--roi_align--roi_layers--2--gradROIAlign--ROIAlignGrad-op4848_output_0_shape_2_256_48_80_kNumberTypeFloat32_DefaultFormat.bin', np.float32)
    dump_grad = dump_grad.reshape(2,256,48,80)
    print('dump compare train')
    print(np.allclose(ms_grad_out_np, dump_grad, atol=0.001, rtol=0.001))

    print(ms_grad_out_np.reshape(-1)[:20])
    print(dump_grad.reshape(-1)[:20])

    torch_out.backward(torch.from_numpy(input_grad).to(device=device))
    # torch_out_grad_np = torch_input.clone().detach().cpu().numpy()
    torch_out_grad_np = torch_input.grad.data.clone().detach().cpu().numpy()

    assert ms_grad_out_np.shape == torch_out_grad_np.shape

    print('torch compare train')
    # print('ms_grad_out', ms_grad_out_np)
    # print('torch_grad_out', torch_out_grad_np)
    print(np.allclose(ms_grad_out_np, torch_out_grad_np, atol=0.001, rtol=0.001))


def test_float16():
    np.random.seed(5)
    data_type = np.float16
    device = torch.device('cuda:0')
    print('data_type', data_type)

    pooled_height = 2
    pooled_width = 2
    spatial_scale = 0.5
    sample_num = 2
    roi_end_mode = 1
    ms_kwargs = dict(pooled_width=pooled_width, pooled_height=pooled_height, spatial_scale=spatial_scale,
                     roi_end_mode=roi_end_mode, sample_num=sample_num)
    torch_kwargs = dict(output_size=(pooled_height, pooled_width), spatial_scale=spatial_scale,
                        sampling_ratio=-1, aligned=True)

    input_shape = (1, 1, 2, 2)
    input_tensor = np.random.rand(*input_shape).astype(data_type)
    rois = np.random.randn(2, 5).astype(data_type)


    ms_input = mindspore.Tensor(input_tensor.copy())
    ms_rois = mindspore.Tensor(rois.copy())

    ms_net = MsNet(**ms_kwargs)
    ms_infer_out = ms_net(ms_input, ms_rois)
    ms_infer_out_np = ms_infer_out.asnumpy()


    torch_input = torch.tensor(input_tensor.copy(), requires_grad=True, device=device)
    torch_boxes = torch.tensor(rois.copy(), requires_grad=True, device=device)
    torch_net = TorchNet(**torch_kwargs).to(device=device).half()
    torch_out = torch_net(torch_input, torch_boxes)
    torch_infer_out_np = torch_out.clone().detach().cpu().numpy()



    assert ms_infer_out_np.shape == torch_infer_out_np.shape
    print('compare infer')
    print(np.allclose(ms_infer_out_np, torch_infer_out_np))

    input_grad = np.random.randn(*ms_infer_out_np.shape).astype(data_type)
    ms_grad_net = MsGrad(ms_net)
    ms_grad_net.set_train()
    ms_grad_out = ms_grad_net(ms_input, ms_rois, mindspore.Tensor(input_grad))

    if isinstance(ms_grad_out, tuple):
        ms_grad_out = ms_grad_out[0]
    ms_grad_out_np = ms_grad_out.asnumpy()

    torch_out.backward(torch.from_numpy(input_grad).to(device=device))
    # torch_out_grad_np = torch_input.clone().detach().cpu().numpy()
    torch_out_grad_np = torch_input.grad.data.clone().detach().cpu().numpy()

    assert ms_grad_out_np.shape == torch_out_grad_np.shape

    print('compare train')
    print(np.allclose(ms_grad_out_np, torch_out_grad_np))


if __name__ == '__main__':
    test_float32()
    #test_float16()


