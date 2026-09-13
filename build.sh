#!/usr/bin/env bash
set -o errexit

python -m pip install --upgrade pip

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python manage.py migrate
python manage.py collectstatic --no-input
