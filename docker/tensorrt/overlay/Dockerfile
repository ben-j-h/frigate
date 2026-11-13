# Frigate amd64 -tensorrt image + the TensorRT runtime, so the ONNX detector's `device: Tensorrt[:<gpu>]` really runs
# TensorRT (FP16 + CUDA graphs) instead of silently falling back to CUDA. Upstream removed TensorRT from the amd64 image
# in #19030 along with the old TensorRT detector.
#
#   docker build -t frigate:0.18.2-tensorrt-trt10 docker/tensorrt/overlay
#   docker build --build-arg FRIGATE_IMAGE=ghcr.io/blakeblackshear/frigate:<tag>-tensorrt --build-arg CUDA_SMS="86 89" ...
#
# TRT_VERSION must be a TensorRT 10 release that the image's onnxruntime-gpu loads (1.24 -> 10.x; tested 10.16.1.11).
# CUDA_SMS = compute capability of every GPU that builds engines: 75 Turing, 80 A100, 86 RTX 30, 89 RTX 40, 120 RTX 50.
ARG FRIGATE_IMAGE=ghcr.io/blakeblackshear/frigate:0.18.2-tensorrt

FROM ${FRIGATE_IMAGE} AS trt
ARG TRT_VERSION=10.16.1.11
ARG CUDA_SMS="86 120"
COPY fetch_trt_libs.py /fetch_trt_libs.py
RUN python3 /fetch_trt_libs.py ${TRT_VERSION} /trt ${CUDA_SMS}

FROM ${FRIGATE_IMAGE}
COPY --from=trt /trt/ /usr/local/lib/tensorrt/
RUN echo /usr/local/lib/tensorrt > /etc/ld.so.conf.d/tensorrt.conf \
    && ldconfig \
    && ldconfig -p | grep -q libnvinfer.so.10 \
    && python3 -c "import ctypes; [ctypes.CDLL(l) for l in ('libnvinfer.so.10', 'libnvonnxparser.so.10', 'libnvinfer_plugin.so.10')]"
