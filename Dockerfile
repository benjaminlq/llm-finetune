FROM nvcr.io/nvidia/pytorch:22.08-py3
WORKDIR /llm
RUN apt-get update && apt-get install -y
COPY ./requirements.txt ./requirements-dev.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --quiet --no-cache-dir \
    && pip install -r requirements-dev.txt --quiet --no-cache-dir
RUN pip install flash-attn

RUN apt-get install -y libaio-dev
# Deep Speed for A100 CUDA Arch Only
RUN git clone https://github.com/microsoft/DeepSpeed/ \
    && cd DeepSpeed && rm -rf build \
    && TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_OPS=1 DS_BUILD_UTILS=1 \
    pip install . --global-option="build_ext" --global-option="-j8" --no-cache -v \
    --disable-pip-version-check 2>&1 | tee build.log

RUN git clone https://github.com/NVIDIA/apex \
    && cd apex \
    && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

RUN git clone https://github.com/microsoft/varuna.git \
    && cd varuna \
    && python setup.py install

ENV JUPYTER_SERVER_ROOT /home/jovyan
ENV NB_PREFIX /
EXPOSE 8888

ENTRYPOINT ["/bin/sh"]

CMD ["-c", "jupyter lab --notebook-dir=${JUPYTER_SERVER_ROOT} --ip=0.0.0.0 --no-browser \
 --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' \
 --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]