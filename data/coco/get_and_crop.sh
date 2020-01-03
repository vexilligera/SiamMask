wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip ./train2017.zip
unzip ./val2017.zip
unzip ./annotations_trainval2017.zip
cd pycocotools && make && cd ..

python par_crop.py --enable_mask --num_threads 64
python gen_json.py
