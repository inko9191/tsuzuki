#include "engine.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>

namespace py = pybind11;

PYBIND11_MODULE(_egemma_native, m) {
    m.doc() = "EmbeddingGemma 2: direct safetensors loading and C++/Metal inference";
    m.def("device_info", &egemma::device_info);
    py::class_<egemma::Engine>(m, "Engine")
        .def(py::init<const std::string&, const std::string&, bool, const std::string&>(),
             py::arg("model_dir"), py::arg("dtype") = "fp32", py::arg("profile") = false, py::arg("matmul") = "auto",
             py::call_guard<py::gil_scoped_release>())
        .def("info", &egemma::Engine::info)
        .def("profile", &egemma::Engine::profile)
        .def("encode", [](egemma::Engine& engine,
                          py::array_t<int32_t, py::array::c_style> ids,
                          py::array_t<int32_t, py::array::c_style> mask,
                          size_t dim, const std::string& capture_dir) {
            if (ids.ndim() != 2 || mask.ndim() != 2 || ids.shape(0) != mask.shape(0) || ids.shape(1) != mask.shape(1))
                throw py::value_error("input_ids and attention_mask must be matching 2D int32 arrays");
            size_t batch=ids.shape(0), length=ids.shape(1);
            std::vector<float> values;
            { py::gil_scoped_release release; values=engine.encode(ids.data(),mask.data(),batch,length,dim,capture_dir); }
            py::array_t<float> result({batch,dim});
            std::memcpy(result.mutable_data(),values.data(),values.size()*sizeof(float));
            return result;
        }, py::arg("input_ids").noconvert(), py::arg("attention_mask").noconvert(),
           py::arg("dim")=768, py::arg("capture_dir")="");
}
