# Use Builtin Datasets

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

Detectron2 has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  coco/
  lvis/
  ILSVRC2012/
  VOC20{07,12}/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

## Expected dataset structure for COCO instance/keypoint detection:

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

You can use the 2014 version of the dataset as well.

Some of the builtin tests (`dev/run_*_tests.sh`) uses a tiny version of the COCO dataset,
which you can download with `./prepare_for_tests.sh`.


## Expected dataset structure for LVIS instance segmentation:
```
coco/
  {train,val,test}2017/
lvis/
  lvis_v0.5_{train,val}.json
  lvis_v0.5_image_info_test.json
  lvis_v1_{train,val}.json
  lvis_v1_image_info_test{,_challenge}.json
```

Install lvis-api by:
```
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

To evaluate models trained on the COCO dataset using LVIS annotations,
run `python prepare_cocofied_lvis.py` to prepare "cocofied" LVIS annotations.

## Expected dataset structure for Pascal VOC:
```
VOC20{07,12}/
  Annotations/
  ImageSets/
    Main/
      trainval.txt
      test.txt
      # train.txt or val.txt, if you use these splits
  JPEGImages/
```

## Expected dataset structure for ILSVRC2012:
Go to [this link](https://www.image-net.org/download-images.php) to download tar files (for training and validation)
```
├── ILSVRC2012_img_train.tar
└── ILSVRC2012_img_val.tar
```
Run bash scripts/extract_ilsvrc.sh for handling the above compressed files.
Be sure if they are arranged like below:
```
./train
├── n07693725
├── ... 
└── n07614500
./val
├── n01440764 
├── ...
└── n04458633
```
Run below scripts for getting json annotations.
```
bash scripts/prepare_ilsvrc.sh datasets/ILSVRC2012/val/ output/temp/ilsvrc_2012_val_info.json datasets/ILSVRC2012/ILSVRC2012_img_val.json tools/ilsvrc2012_classes_name.txt datasets/ILSVRC2012/ILSVRC2012_img_val_converted.json

bash scripts/prepare_ilsvrc.sh datasets/ILSVRC2012/train/ output/temp/ilsvrc_2012_train_info.json datasets/ILSVRC2012/ILSVRC2012_img_train.json tools/ilsvrc2012_classes_name.txt datasets/ILSVRC2012/ILSVRC2012_img_train_converted.json
```

```
ILSCRC2012/
  ILSVRC2012_img_train_converted.json
  ILSVRC2012_img_val_converted.json
  {train,val}/
    n01440764/*.JPEG # image files that are mentioned in the corresponding json
    ......
    n15075141/*.JPEG # image files that are mentioned in the corresponding json
    
```