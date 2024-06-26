# diffusers를 활용한 파인툰

파이썬 3.10 버전으로 NVIDIA GPU(4090)를 사용하여 diffusers를 활용한 파인튜닝을 수행합니다.

## 로라

### 로라 샘플 코드
- [train_text_to_image_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)

### 드림부스 페이지에서 로라를 사용하는 방법을 설명한 것
- [train_dreambooth_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py)

## 드림부스
- [train_dreambooth.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)

## 설치

setup.sh 실행

파이썬은 3.10 버전을 사용합니다.

### 로컬 기동

```shell
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

accelerate config default
```

Optional: install [xformers](https://huggingface.co/docs/diffusers/optimization/xformers) and
add `--enable_xformers_memory_efficient_attention`

```shell
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

## 실행

### 텐서보드

```shell
tensorboard --logdir=models/dreambooth-lora/dog/logs --bind_all
tensorboard --logdir=models/dreambooth-lora/sunglasses/logs --bind_all
tensorboard --logdir=models/lora/miles/logs --bind_all

```

```shell
python src/convert-to-safetensors.py --file models/lora/miles/checkpoint-1000 --output models/lora/miles/checkpoint-1000-safetensors
```

```shell
python src/generate-lora.py --prompt "A photo of a cat in a bucket" --steps 200 --model_path /repositories/diffusers_finetune/models/lora/miles --output_folder outputs/lora/miles
```

```shell
python src/convert_diffusers_to_original_stable_diffusion.py --model_path /repositories/anything-v5 --checkpoint_path  /repositories/dog.safetensors --use_safetensors --half
```


# 참고

-----

* [diffusers](https://github.com/huggingface/diffusers)
* [accelerate](https://github.com/huggingface/accelerate)
* [finetune-sd](https://github.com/harrywang/finetune-sd)
* [Stuck Process and SIGTERM Signal Interruption During Training with accelerate launch](https://github.com/hiyouga/LLaMA-Factory/issues/2359)
* [nvidia nccl Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
* [dreambooth-training-guide](https://github.com/nitrosocke/dreambooth-training-guide)
* [dreambooth_classifiers_and_instanceclass_prompts](https://www.reddit.com/r/StableDiffusion/comments/z4s07l/dreambooth_classifiers_and_instanceclass_prompts/)
* [ELI5 Training](https://github.com/d8ahazard/sd_dreambooth_extension/wiki/ELI5-Training)
* [Dreambooth로 특정 화가의 스타일 학습시키기 3편](https://www.clien.net/service/board/cm_aigurim/18160386)
* [waifu-diffusion](https://huggingface.co/hakurei/waifu-diffusion)
* [calliope-legacy](https://huggingface.co/NovelAI/calliope-legacy)
