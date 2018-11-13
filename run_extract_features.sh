#!/bin/bash

echo 'python3 extract_features.py --image_paths=500K_imagenet/imagepaths_val.txt --save_file=500K_imagenet/val.pkl --batch_size=1000'
python3 extract_features.py --image_paths=500K_imagenet/imagepaths_val.txt --save_file=500K_imagenet/val.pkl --batch_size=1000
echo ''
echo ''


echo 'python3 extract_features.py --image_paths=500K_imagenet/imagepaths_test.txt --save_file=500K_imagenet/test.pkl --batch_size=1000'
python3 extract_features.py --image_paths=500K_imagenet/imagepaths_test.txt --save_file=500K_imagenet/test.pkl --batch_size=1000
echo ''
echo ''


echo 'python3 extract_features.py --image_paths=500K_imagenet/imagepaths_train.txt --save_file=500K_imagenet/train.pkl --batch_size=1000'
python3 extract_features.py --image_paths=500K_imagenet/imagepaths_train.txt --save_file=500K_imagenet/train.pkl --batch_size=1000
echo ''
echo ''


