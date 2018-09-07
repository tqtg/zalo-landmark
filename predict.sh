#/bin/bash

data_dir=$1

rm -rf submissions
mkdir submissions

export CUDA_VISIBLE_DEVICES=0

echo '0_inception_resnet_v2'
python3 src/predict.py --data_dir $data_dir --net 'inception_resnet_v2' --checkpoint_dir 'models/0_inception_resnet_v2' --batch_size 10 --image_size 299

echo '1_inception_resnet_v2'
python3 src/predict.py --data_dir $data_dir --net 'inception_resnet_v2' --checkpoint_dir 'models/1_inception_resnet_v2' --batch_size 10 --image_size 299

echo '2_resnet_v2_152'
python3 src/predict.py --data_dir $data_dir --net 'resnet_v2_152' --checkpoint_dir 'models/2_resnet_v2_152' --batch_size 10 --image_size 299

for i in {0..9}
do
    printf "\ninception_resnet_v2_fold_$i\n"
    python3 src/predict.py --data_dir $data_dir --net 'inception_resnet_v2' --checkpoint_dir "models/inception_resnet_v2_fold$i" --batch_size 10 --image_size 299
done

python3 src/prepare_submission.py
