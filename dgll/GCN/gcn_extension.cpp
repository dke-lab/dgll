// gcn_extension.cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>

extern "C" void launch_gcn_fused_kernel(
    const int* row_ptr, const int* col_idx, const float* values,
    const float* X, const float* W, float* H,
    const int* num_neighbors,
    int N, int F_padded, int actual_F, int H_dim, int total_nnz
);

extern "C" void launch_gcn_fused_kernel_backward_optimized(
    const int* row_ptr, const int* col_idx, const float* values,
    const float* X, const float* W, const float* grad_output,
    float* grad_W, float* grad_X,
    const int* num_neighbors,
    int N, int F_padded, int actual_F, int H_dim, int total_nnz
);

namespace py = pybind11;

torch::Tensor gcn_fused_forward(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor num_neighbors,
    int actual_F
) {
    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be a CUDA tensor");
    TORCH_CHECK(col_idx.is_cuda(), "col_idx must be a CUDA tensor");
    TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(num_neighbors.is_cuda(), "num_neighbors must be a CUDA tensor");

    int N = X.size(0);
    int F_padded = X.size(1);
    int H_dim = W.size(1);
    int total_nnz = col_idx.size(0);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor H = torch::zeros({N, H_dim}, options);

    launch_gcn_fused_kernel(
        row_ptr.data_ptr<int>(),
        col_idx.data_ptr<int>(),
        values.data_ptr<float>(),
        X.data_ptr<float>(),
        W.data_ptr<float>(),
        H.data_ptr<float>(),
        num_neighbors.data_ptr<int>(),
        N, F_padded, actual_F, H_dim, total_nnz
    );

    return H;
}

std::vector<torch::Tensor> gcn_fused_backward(
    torch::Tensor grad_output,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor num_neighbors,
    int actual_F
) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be a CUDA tensor");
    TORCH_CHECK(col_idx.is_cuda(), "col_idx must be a CUDA tensor");
    TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(num_neighbors.is_cuda(), "num_neighbors must be a CUDA tensor");

    int N = X.size(0);
    int F_padded = X.size(1);
    int H_dim = W.size(1);
    int total_nnz = col_idx.size(0);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor grad_W = torch::zeros_like(W);
    torch::Tensor grad_X = torch::zeros_like(X);

    launch_gcn_fused_kernel_backward_optimized(
        row_ptr.data_ptr<int>(),
        col_idx.data_ptr<int>(),
        values.data_ptr<float>(),
        X.data_ptr<float>(),
        W.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_W.data_ptr<float>(),
        grad_X.data_ptr<float>(),
        num_neighbors.data_ptr<int>(),
        N, F_padded, actual_F, H_dim, total_nnz
    );

    return {grad_X, grad_W}; // Return gradients for X and W
}

PYBIND11_MODULE(gcn_extension, m) {
    m.def("gcn_fused_forward", &gcn_fused_forward, "GCN forward pass",
          py::arg("row_ptr"), py::arg("col_idx"), py::arg("values"),
          py::arg("X"), py::arg("W"), py::arg("num_neighbors"), py::arg("actual_F"));
    m.def("gcn_fused_backward", &gcn_fused_backward, "GCN backward pass",
          py::arg("grad_output"), py::arg("row_ptr"), py::arg("col_idx"), py::arg("values"),
          py::arg("X"), py::arg("W"), py::arg("num_neighbors"), py::arg("actual_F"));
}