wget http://image-net.org/image/ILSVRC2015/ILSVRC2015_DET.tar.gz
tar -xzvf ./ILSVRC2015_DET.tar.gz

python par_crop.py 511 64
python gen_json.py
