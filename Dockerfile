FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
LABEL authors="beni"
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

RUN apt-get update && apt-get install -y git && apt-get clean

RUN pip install jupyter jupyter_http_over_ws
RUN jupyter server extension enable --py jupyter_http_over_ws

# Install diffusers
RUN pip install diffusers accelerate transformers

WORKDIR /workspace
EXPOSE 8888

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/workspace --ip 0.0.0.0 --no-browser --allow-root"]
