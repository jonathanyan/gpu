import numpy as np
import mindspore

from mindspore import context
import mindspore.ops.operations as P
import mindspore.nn as nn
from mindspore.ops import composite as C


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




def test_float32():
    np.random.seed(5)
    data_type = np.float32
    print('data_type', data_type)


    pooled_height = 7
    pooled_width = 7
    # spatial_scale = 1.0/16.0
    spatial_scale = 1.0
    sample_num = 2 
    roi_end_mode = 1
    ms_kwargs = dict(pooled_width=pooled_width, pooled_height=pooled_height, spatial_scale=spatial_scale,
                     roi_end_mode=roi_end_mode, sample_num=sample_num)


    # input_tensor =  np.ones(49, dtype=np.float32) * 0.123456789
    input_tensor =  np.ones(48*80, dtype=np.float32) * 0.123456789
    rois = np.array([0, 0.5, 0.7, 13.2, 24.3], dtype=np.float32)* 1.123456789
    input_tensor = input_tensor.reshape(1,1,48,80)
    rois = rois.reshape(1,5)


    ms_input = mindspore.Tensor(input_tensor.copy())
    ms_rois = mindspore.Tensor(rois.copy())

    ms_net = MsNet(**ms_kwargs)
    ms_infer_out = ms_net(ms_input, ms_rois)
    ms_infer_out_np = ms_infer_out.asnumpy()

    print('ms_infer_out', ms_infer_out_np)



if __name__ == '__main__':
    test_float32()


