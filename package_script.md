# Script for creating a new venv

```bash
python3 -m venv ./venv
. ./venv/bin/activate

pip install --upgrade pip
pip list
pip install torch torchvision
pip install scikit-learn tqdm jupyterlab matplotlib pandas
pip install openpyxl

sudo apt-get install python-opencv
```
