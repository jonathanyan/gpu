/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARGMAXWITHVALUEGPUKERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARGMAXWITHVALUEGPUKERNEL_H_

#include <map>
#include <string>
#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
const std::map<std::string, cudnnReduceTensorOp_t> kReduceTypeMap = {
  {"ArgMaxWithValue", CUDNN_REDUCE_TENSOR_MAX},
};
template <typename T, typename S>
class ArgmaxWithValueGpuKernel : public GpuKernel {
 public:
  ArgmaxWithValueGpuKernel()
      : cudnn_handle_(nullptr),
        reduce_tensor_op_(CUDNN_REDUCE_TENSOR_MAX),
        data_type_(CUDNN_DATA_FLOAT),
        nan_prop_(CUDNN_NOT_PROPAGATE_NAN),
        reduce_indices_(CUDNN_REDUCE_TENSOR_FLATTENED_INDICES),
        reduce_tensor_descriptor_(nullptr),
        inputA_descriptor_(nullptr),
        indexB_descriptor_(nullptr),
        outputC_descriptor_(nullptr),
        keep_dims_(false),
        all_match_(false),
        is_null_input_(false),
        input_size_(0),
        index_size_(0),
        output_size_(0),
        workspace_size_(0) {}
  ~ArgmaxWithValueGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    S *index_addr = GetDeviceAddress<S>(outputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 1);
    T *workspace_addr = GetDeviceAddress<T>(workspace, 0);

    const float alpha = 1;
    const float beta = 0;
    if (all_match_) {
      MS_LOG(WARNING)
        << "The corresponding dimensions of the input and output tensors all match. No need to call cuDNN kernel.";
      CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(output_addr, input_addr, inputs[0]->size, cudaMemcpyDeviceToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync failed in ArgmaxWithValueGpuKernel::Launch.");
    } else {
      MS_LOG(WARNING) << "cudnn argmaxwithvalue involved.";
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, index_addr, index_size_,
                                                    workspace_addr, workspace_size_, &alpha, inputA_descriptor_,
                                                    input_addr, &beta, outputC_descriptor_, output_addr),
                                  "cudnnReduceTensor failed.");
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but reduce op needs 1 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 2) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but reduce op needs 2 output.";
      return false;
    }
    int input_dim_length = SizeToInt(AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0).size());

    if (AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("axis")->isa<ValueTuple>() ||
        AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("axis")->isa<ValueList>()) {
      auto attr_axis = GetAttr<std::vector<int>>(kernel_node, "axis");
      if (attr_axis.empty()) {
        axis_.push_back(-1);
      } else {
        for (auto axis : attr_axis) {
          axis < 0 ? axis_.push_back(axis + input_dim_length) : axis_.push_back(axis);
        }
      }
    } else if (AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("axis")->isa<Int32Imm>()) {
      int axis = GetAttr<int>(kernel_node, "axis");
      axis < 0 ? axis_.push_back(axis + input_dim_length) : axis_.push_back(axis);
    } else {
      MS_LOG(EXCEPTION) << "Attribute axis type is invalid.";
    }

    keep_dims_ = GetAttr<bool>(kernel_node, "keep_dims");

    auto inputA_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto indexB_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    auto outputC_shape = AnfAlgo::GetOutputInferShape(kernel_node, 1);
    is_null_input_ = CHECK_NULL_INPUT(inputA_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "ArgmaxWithValueGpuKernel input is null";
      InitSizeLists();
      return true;
    }
    InferInAndOutDesc(inputA_shape, indexB_shape, outputC_shape);
    InferArgmaxWithValueType(kernel_node);

    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateReduceTensorDescriptor(&reduce_tensor_descriptor_),
                                "cudnnCreateReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&inputA_descriptor_),
                                "cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&indexB_descriptor_),
                                "cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&outputC_descriptor_),
                                "cudnnCreateTensorDescriptor failed.");
  }
  void InitSizeLists() override {
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(inputA_descriptor_, &input_size_),
                                "cudnnGetTensorSizeInBytes failed.");
    input_size_list_.push_back(input_size_);

    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(indexB_descriptor_, &index_size_),
                                "cudnnGetTensorSizeInBytes failed.");
    output_size_list_.push_back(index_size_);

    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(outputC_descriptor_, &output_size_),
                                "cudnnGetTensorSizeInBytes failed.");
    output_size_list_.push_back(output_size_);

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnGetReductionWorkspaceSize(cudnn_handle_, reduce_tensor_descriptor_, inputA_descriptor_, outputC_descriptor_,
                                     &workspace_size_),
      "cudnnGetReductionWorkspaceSize failed.");
    workspace_size_list_.push_back(workspace_size_);
    return;
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyReduceTensorDescriptor(reduce_tensor_descriptor_),
                               "cudnnDestroyReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(inputA_descriptor_),
                               "cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(indexB_descriptor_),
                               "cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(outputC_descriptor_),
                               "cudnnDestroyTensorDescriptor failed.");
  }
  void InferArgmaxWithValueType(const CNodePtr &kernel_node) {
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    auto iter = kReduceTypeMap.find(kernel_name);
    if (iter == kReduceTypeMap.end()) {
      MS_LOG(EXCEPTION) << "Array reduce kernel type " << kernel_name << " is not supported.";
    } else {
      reduce_tensor_op_ = iter->second;
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetReduceTensorDescriptor(reduce_tensor_descriptor_, reduce_tensor_op_, CUDNN_DATA_FLOAT, nan_prop_,
                                     reduce_indices_, CUDNN_32BIT_INDICES),
      "cudnnSetReduceTensorDescriptor failed");
    return;
  }
  void InferInAndOutDesc(const std::vector<size_t> &input_shape, const std::vector<size_t> &index_shape,
                         const std::vector<size_t> &output_shape) {
    std::vector<int> inputA;
    std::vector<size_t> indexB_shape = index_shape;
    std::vector<size_t> outputC_shape = output_shape;
    const int split_dim = 4;

    if (input_shape.size() <= split_dim) {
      ShapeNdTo4d(input_shape, &inputA);
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensor4dDescriptor(inputA_descriptor_, CUDNN_TENSOR_NCHW, data_type_,
                                                             inputA[0], inputA[1], inputA[2], inputA[3]),
                                  "cudnnSetTensor4dDescriptor failed");
    } else {
      CudnnSetTensorNdDescriptor(input_shape, inputA_descriptor_, data_type_);
      for (auto dim : input_shape) {
        inputA.emplace_back(SizeToInt(dim));
      }
    }

    std::vector<int> indexB;
    if (indexB_shape.size() <= split_dim) {
      ShapeNdTo4d(indexB_shape, &indexB);
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensor4dDescriptor(indexB_descriptor_, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT32,
                                                             indexB[0], indexB[1], indexB[2], indexB[3]),
                                  "cudnnSetTensor4dDescriptor failed");
    } else {
      CudnnSetTensorNdDescriptor(indexB_shape, indexB_descriptor_, data_type_);
      for (auto dim : indexB_shape) {
        indexB.emplace_back(SizeToInt(dim));
      }
    }

    if (axis_[0] == -1) {
      outputC_shape.resize(input_shape.size(), 1);
      if (outputC_shape.size() <= split_dim) {
        CHECK_CUDNN_RET_WITH_EXCEPT(
          cudnnSetTensor4dDescriptor(outputC_descriptor_, CUDNN_TENSOR_NCHW, data_type_, 1, 1, 1, 1),
          "cudnnSetTensor4dDescriptor failed");
      } else {
        CudnnSetTensorNdDescriptor(outputC_shape, outputC_descriptor_, data_type_);
      }

      for (auto dim : inputA) {
        if (dim != 1) {
          return;
        }
      }

      all_match_ = true;
      return;
    }

    std::vector<int> outputC;
    if (!keep_dims_) {
      for (auto i : axis_) {
        (void)(outputC_shape.insert(outputC_shape.begin() + i, 1));
      }
    }

    if (outputC_shape.size() <= split_dim) {
      ShapeNdTo4d(outputC_shape, &outputC);
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensor4dDescriptor(outputC_descriptor_, CUDNN_TENSOR_NCHW, data_type_,
                                                             outputC[0], outputC[1], outputC[2], outputC[3]),
                                  "cudnnSetTensor4dDescriptor failed");
    } else {
      CudnnSetTensorNdDescriptor(outputC_shape, outputC_descriptor_, data_type_);
      for (auto dim : outputC_shape) {
        outputC.emplace_back(SizeToInt(dim));
      }
    }

    if (inputA == outputC) {
      all_match_ = true;
    }
    return;
  }

  cudnnHandle_t cudnn_handle_;
  cudnnReduceTensorOp_t reduce_tensor_op_;
  cudnnDataType_t data_type_;
  cudnnNanPropagation_t nan_prop_;
  cudnnReduceTensorIndices_t reduce_indices_;
  cudnnReduceTensorDescriptor_t reduce_tensor_descriptor_;
  cudnnTensorDescriptor_t inputA_descriptor_;
  cudnnTensorDescriptor_t indexB_descriptor_;
  cudnnTensorDescriptor_t outputC_descriptor_;

  std::vector<int> axis_;
  bool keep_dims_;
  bool all_match_;
  bool is_null_input_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  size_t input_size_;
  size_t index_size_;
  size_t output_size_;
  size_t workspace_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARGMAXWITHVALUEGPUKERNEL_H_
