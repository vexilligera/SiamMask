wget http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz
tar -xzvf ./ILSVRC2015_VID.tar.gz
mv ILSVRC2015/Annotations/VID/val ILSVRC2015/Annotations/VID/train/
mv ILSVRC2015/Data/VID/val ILSVRC2015/Data/VID/train/

python parse_vid.py
python par_crop.py 511 64
python gen_json.py
