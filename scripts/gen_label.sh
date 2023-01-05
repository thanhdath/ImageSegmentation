wget http://download.tensorflow.org/models/deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz
tar xvf deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz

mkdir data/images
echo "Place images in folder data/images"

python segmentation.py 
python gen_annotation_for_labelme.py

echo "Pre-annotation files in output/for_labeling"
