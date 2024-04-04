from pathlib import Path

from train_util import glob_images_pathlib


def test_glob_images_pathlib():
  path = Path("../data/dreambooth")
  actual = glob_images_pathlib(path, recursive=True)
  assert len(actual) == 84
