# diffusers를 활용한 파인툰

파이썬 3.10 버전으로 NVIDIA GPU(4090)를 사용하여 diffusers를 활용한 파인튜닝을 수행합니다.

- [train_text_to_image_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)
- [train_dreambooth_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py)
- [train_dreambooth.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)

## 설치

### 로컬 기동

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
accelerate config default
```
Optional: install [xformers](https://huggingface.co/docs/diffusers/optimization/xformers) and add `--enable_xformers_memory_efficient_attention`

```
pip install xformers
```

```shell
deactivate
```


### 도커 기동

```shell
docker build -t pytorch-jupyter .

docker run --gpus all -d --rm -p 8888:8888 pytorch-jupyter


docker run --gpus all -d tensorflow/tensorflow:2.15.0-gpu-jupyter

docker run --gpus all -d pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

docker run --gpus all -it pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel /bin/bash

docker run --gpus all -it pytorch-jupyter:latest /bin/bash

docker exec -it 430e2e8bd9f4 /bin/bash

docker logs --tail 100 -f 430e2e8bd9f4
```


# 참고

-----

* [diffusers](https://github.com/huggingface/diffusers)
* [accelerate](https://github.com/huggingface/accelerate)
* [finetune-sd](https://github.com/harrywang/finetune-sd)