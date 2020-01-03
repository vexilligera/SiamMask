python download_from_gdrive.py https://drive.google.com/uc?id=18S_db1cFgSD1RsMsofJLkd6SyR9opk6a --output train.zip
unzip ./train.zip
pip install opencv-python
python parse_ytb_vos.py

python par_crop.py 511 64
python gen_json.py
