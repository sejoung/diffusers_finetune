from safetensors.torch import save_file

from convert_diffusers_to_original_stable_diffusion import get_state_dict


def test_convert_to_pathlib():
  state_dict = get_state_dict("/Users/beni/anything-v5/")
  save_file(state_dict, "/Users/beni/Downloads/diffusers_model_original.safetensors")
