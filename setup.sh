#!/usr/bin/bash

[ -d venv ] || python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

accelerate config default

deactivate
