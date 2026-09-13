#!/usr/bin/env bash
set -o errexit

python -m pip install --upgrade pip

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements.txt

python manage.py migrate
python manage.py collectstatic --no-input