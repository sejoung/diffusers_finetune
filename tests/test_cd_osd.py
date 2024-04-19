from diffusers import DiffusionPipeline
from safetensors.torch import save_file

from convert_diffusers_to_original_stable_diffusion import get_state_dict


def test_convert_to_pathlib():
  pipeline = DiffusionPipeline.from_pretrained(
    "stablediffusionapi/anything-v5",
    torch_dtype="fp32",
  )
  output = "/Users/beni/Downloads/diffusers_model_original"
  pipeline.save_pretrained(output)
  state_dict = get_state_dict(output)
  save_file(state_dict, "/Users/beni/Downloads/anything-v5.safetensors")
