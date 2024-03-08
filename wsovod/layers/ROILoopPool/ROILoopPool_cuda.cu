#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>

#include "cuda_helpers.h"

template <typename T>
__global__ void RoILoopPoolForward(
    const int nthreads,
    const T* input,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const T* rois,
    T* output,
    int* argmax_data,
    const int num_rois,
    const float context_ratio_) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // x1 y1 x2 y2
    float x1 = offset_rois[1];
    float y1 = offset_rois[2];
    float x2 = offset_rois[3];
    float y2 = offset_rois[4];

    float rois_w = x2 - x1;
    float rois_h = y2 - y1;

    float rois_inner_w = rois_w / context_ratio_;
    float rois_inner_h = rois_h / context_ratio_;

    float rois_outer_w = rois_w * context_ratio_;
    float rois_outer_h = rois_h * context_ratio_;

    float inner_residual_w = rois_w - rois_inner_w;
    float inner_residual_h = rois_h - rois_inner_h;

    float outer_residual_w = rois_outer_w - rois_w;
    float outer_residual_h = rois_outer_h - rois_h;

    float x1_inner = x1 + inner_residual_w / 2;
    float y1_inner = y1 + inner_residual_h / 2;
    float x2_inner = x2 - inner_residual_w / 2;
    float y2_inner = y2 - inner_residual_h / 2;

    float x1_outer = x1 - outer_residual_w / 2;
    float y1_outer = y1 - outer_residual_h / 2;
    float x2_outer = x2 + outer_residual_w / 2;
    float y2_outer = y2 + outer_residual_h / 2;

    x1_inner = min(max(x1_inner, T(0)), T(1.0 * width / spatial_scale));
    y1_inner = min(max(y1_inner, T(0)), T(1.0 * height / spatial_scale));
    x2_inner = min(max(x2_inner, T(0)), T(1.0 * width / spatial_scale));
    y2_inner = min(max(y2_inner, T(0)), T(1.0 * height / spatial_scale));

    x1_outer = min(max(x1_outer, T(0)), T(1.0 * width / spatial_scale));
    y1_outer = min(max(y1_outer, T(0)), T(1.0 * height / spatial_scale));
    x2_outer = min(max(x2_outer, T(0)), T(1.0 * width / spatial_scale));
    y2_outer = min(max(y2_outer, T(0)), T(1.0 * height / spatial_scale));

    {
      // outer rectangle of the region
      int roi_start_w = round(offset_rois[1] * spatial_scale);
      int roi_start_h = round(offset_rois[2] * spatial_scale);
      int roi_end_w = round(offset_rois[3] * spatial_scale);
      int roi_end_h = round(offset_rois[4] * spatial_scale);

      // inner rectangle of the region
      int roi_start_w_in = round(x1_inner * spatial_scale);
      int roi_start_h_in = round(y1_inner * spatial_scale);
      int roi_end_w_in = round(x2_inner * spatial_scale);
      int roi_end_h_in = round(y2_inner * spatial_scale);

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);
      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

      int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
      int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
      int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
      int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart + roi_start_h, 0), height);
      hend = min(max(hend + roi_start_h, 0), height);
      wstart = min(max(wstart + roi_start_w, 0), width);
      wend = min(max(wend + roi_start_w, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Define an empty pooling region to be zero
      // T maxval = is_empty ? 0 : -FLT_MAX;
      // assum all input is >=0
      T maxval = 0;
      T maxval_F = 0;
      // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
      int maxidx = -1;
      int maxidx_F = -1;
      const T* offset_input =
          input + (roi_batch_ind * channels + c) * height * width;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int input_index = h * width + w;
          if (offset_input[input_index] > maxval) {
            maxval = offset_input[input_index];
            maxidx = input_index;
          }

          const bool is_inside_h = h > roi_start_h_in && h < roi_end_h_in;
          const bool is_inside_w = w > roi_start_w_in && w < roi_end_w_in;
          if (is_inside_h && is_inside_w) {
            continue;
          }
          // if it is not inside the inner rectangle of the region
          if (offset_input[input_index] > maxval_F) {
            maxval_F = offset_input[input_index];
            maxidx_F = input_index;
          }
        }
      }
      output[index] = maxval;
      argmax_data[index] = maxidx;

      int frame_offset = num_rois * channels * pooled_width * pooled_height;
      output[index + frame_offset] = maxval_F;
      argmax_data[index + frame_offset] = maxidx_F;
    }

    {
      // outer rectangle of the region
      int roi_start_w = round(x1_outer * spatial_scale);
      int roi_start_h = round(y1_outer * spatial_scale);
      int roi_end_w = round(x2_outer * spatial_scale);
      int roi_end_h = round(y2_outer * spatial_scale);

      // inner rectangle of the region
      int roi_start_w_in = round(offset_rois[1] * spatial_scale);
      int roi_start_h_in = round(offset_rois[2] * spatial_scale);
      int roi_end_w_in = round(offset_rois[3] * spatial_scale);
      int roi_end_h_in = round(offset_rois[4] * spatial_scale);

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);
      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

      int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
      int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
      int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
      int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart + roi_start_h, 0), height);
      hend = min(max(hend + roi_start_h, 0), height);
      wstart = min(max(wstart + roi_start_w, 0), width);
      wend = min(max(wend + roi_start_w, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Define an empty pooling region to be zero
      // T maxval = is_empty ? 0 : -FLT_MAX;
      // assum all input is >=0
      T maxval = 0;
      // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
      int maxidx = -1;
      const T* offset_input =
          input + (roi_batch_ind * channels + c) * height * width;
      for (int h = hstart; h < hend; ++h) {
        const bool is_inside_h = h > roi_start_h_in && h < roi_end_h_in;
        for (int w = wstart; w < wend; ++w) {
          const bool is_inside_w = w > roi_start_w_in && w < roi_end_w_in;
          if (is_inside_h && is_inside_w) {
            continue;
          }
          // if it is not inside the inner rectangle of the region
          int input_index = h * width + w;
          if (offset_input[input_index] > maxval) {
            maxval = offset_input[input_index];
            maxidx = input_index;
          }
        }
      }
      int context_offset =
          2 * num_rois * channels * pooled_width * pooled_height;
      output[index + context_offset] = maxval;
      argmax_data[index + context_offset] = maxidx;
    }
  }
}

template <typename T>
__global__ void RoILoopPoolBackward(
    const int nthreads,
    const T* grad_output,
    const int* argmax_data,
    const int num_rois,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    T* grad_input,
    const T* rois,
    const int n_stride,
    const int c_stride,
    const int h_stride,
    const int w_stride) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + (n % num_rois) * 5;
    int roi_batch_ind = offset_rois[0];
    T* grad_input_offset =
        grad_input + ((roi_batch_ind * channels + c) * height * width);

    int output_offset = n * n_stride + c * c_stride;
    const int* argmax_data_offset =
        argmax_data + (n * channels + c) * pooled_height * pooled_width;
    int argmax = argmax_data_offset[ph * pooled_width + pw];

    if (argmax != -1) {
      atomicAdd(
          grad_input_offset + argmax,
          static_cast<T>(
              grad_output[output_offset + ph * h_stride + pw * w_stride]));
    }
  }
}

namespace wsovod {

std::tuple<at::Tensor, at::Tensor> ROILoopPool_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width) {
  AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.is_cuda(), "rois must be a CUDA tensor");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "ROILoopPool_forward_cuda";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});

  at::cuda::CUDAGuard device_guard(input.device());

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  at::Tensor output = at::zeros(
      {num_rois * 3, channels, pooled_height, pooled_width}, input.options());
  at::Tensor argmax = at::zeros(
      {num_rois * 3, channels, pooled_height, pooled_width},
      input.options().dtype(at::kInt));

  auto output_size = num_rois * pooled_height * pooled_width * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  if (output.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(output, argmax);
  }

  auto input_ = input.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "ROILoopPool_forward", [&] {
        RoILoopPoolForward<scalar_t><<<grid, block, 0, stream>>>(
            output_size,
            input_.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            rois_.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            argmax.data_ptr<int>(),
            num_rois,
            1.8);
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(output, argmax);
}

at::Tensor ROILoopPool_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width) {
  // Check if input tensors are CUDA tensors
  AT_ASSERTM(grad.is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.is_cuda(), "rois must be a CUDA tensor");
  AT_ASSERTM(argmax.is_cuda(), "argmax must be a CUDA tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2},
      argmax_t{argmax, "argmax", 3};

  at::CheckedFrom c = "ROILoopPool_backward_cuda";
  at::checkAllSameGPU(c, {grad_t, rois_t, argmax_t});
  at::checkAllSameType(c, {grad_t, rois_t});

  at::cuda::CUDAGuard device_guard(grad.device());

  auto num_rois = rois.size(0);

  at::Tensor grad_input =
      at::zeros({batch_size, channels, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_input;
  }

  int n_stride = grad.stride(0);
  int c_stride = grad.stride(1);
  int h_stride = grad.stride(2);
  int w_stride = grad.stride(3);

  auto argmax_ = argmax.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "ROILoopPool_backward", [&] {
        RoILoopPoolBackward<scalar_t><<<grid, block, 0, stream>>>(
            grad.numel(),
            grad.data_ptr<scalar_t>(),
            argmax_.data_ptr<int>(),
            num_rois,
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            grad_input.data_ptr<scalar_t>(),
            rois_.data_ptr<scalar_t>(),
            n_stride,
            c_stride,
            h_stride,
            w_stride);
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}

} // namespace wsovod
