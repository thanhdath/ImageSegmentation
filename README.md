# ImageSegmentation

### Create environment
```
conda create -n sia python=3.9
conda activate sia
pip install -r requirements.txt

# Download pretrained model
wget http://download.tensorflow.org/models/deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz
tar xvf deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz
```

### Run API
```
python api.py
```

### Test how API works
```
python test.py
```
