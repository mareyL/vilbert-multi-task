

# 12-in-1: Multi-Task Vision and Language Representation Learning

Please cite the following if you use this code. Code and pre-trained models for [12-in-1: Multi-Task Vision and Language Representation Learning](http://openaccess.thecvf.com/content_CVPR_2020/html/Lu_12-in-1_Multi-Task_Vision_and_Language_Representation_Learning_CVPR_2020_paper.html):

```
@InProceedings{Lu_2020_CVPR,
author = {Lu, Jiasen and Goswami, Vedanuj and Rohrbach, Marcus and Parikh, Devi and Lee, Stefan},
title = {12-in-1: Multi-Task Vision and Language Representation Learning},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

and [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265):

```
@inproceedings{lu2019vilbert,
  title={Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks},
  author={Lu, Jiasen and Batra, Dhruv and Parikh, Devi and Lee, Stefan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={13--23},
  year={2019}
}
```

## Repository Setup

1. Create a fresh conda environment, and install all dependencies.

```text
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
git clone --recursive https://github.com/facebookresearch/vilbert-multi-task.git
cd vilbert-multi-task
pip install -r requirements.txt
```

2. Install pytorch
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch #Install cudatoolkit that fits the computer version , same as nvcc --version
```

3. Install apex, follows https://github.com/NVIDIA/apex

4. Install this codebase as a package in this environment.
```text
python setup.py develop
```
5. Install gitmodules with 
```text
 git submodule init
 git submodule update
cd vilbert-multi-task/tools/refer
python setup.py install
make
#Then replace refer.py byt https://gist.github.com/vedanuj/9d3497d107cfca0b6f3dfdc28d5cb226 to update from Python2 version to Python3
``` 

# Moviescope dataset classification

## Dataset
### Download data
To download the moviescope dataset, use the ressources available in data/moviescope, to download the videos from the csv file put in the same folder:

```
sh data/moviescope/download.sh
```
This will extract all the videos in webm format (for the frame extraction) in a trailer folder.

To extract the plots from the pickles:
```
python data/moviescope/makeCSV.py rawdata_train.p <path_to_made_csv> <path_to_made_ground_truth_csv>
```
This truncate the text at the 511th characters, could be changed.


### Extract Frames from Video
Use this script to extract frames from the video.
```
python script/ME/extract_frames.py --output_folder <output_folder> --video_dir <video_dir> --frames <frames>
```
Use the `frames` parameter for the number of frames to be extracted (default is 1 i.e., the middle frame of the video). The extracted frames are saved as `<output_folder>/<video-id>_<frame_count>.jpg` where `<frame_count>` in `[0..<frames>-1]` (and `<output_folder>/<video-id>.jpg` when extracting only one frame). Keep this structure since it is used by the `script/ME/average_features.py` or `script/extract_features.py` scripts.
Make sure to have writing permission for the `output_folder`. Otherwise, here is an example to use
```
sudo /home/<user>/miniconda3/envs/vilbert-mt/bin/python script/ME/extract_frames.py --output_folder /MediaEval/dev-set/source_output --video_dir /MediaEval/dev-set/sources --frames 1 
```

### Extract Features for Multiple Frames
Use `script/extract_features.py` and add `samples` parameter for the number of frames to use.
```
python script/extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --image_dir datasets/ME/images/train --output_folder datasets/ME/features_100/ME_trainval_resnext152_faster_rcnn_genome.lmdb/ --samples 5
```

### Average Visual Feature Vectors
If using multiple extracted frames from each video, this script is used to average already extracted features. Features files should be named `<video-id>_<feature_count>.npy` where `<feature_count>` in `[0..<feature_number>]`.
```
python script/ME/average_features.py --features_dir <path_to_directory_with_features> --output_folder <path_to_output_averaged_features>
```
### Convert Visual Feature Vectors to lmdb
```
python script/convert_to_lmdb.py  --features_dir <path_to_directory_with_features> --lmdb_file  <path_to_output_lmdb_file>
```

## Visiolinguistic Pre-training and Multi Task Training

### Pretraining on Conceptual Captions

```
python train_concap.py --bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json --train_batch_size 512 --objective 1 --file_path <path_to_extracted_cc_features>
```
[Download link](https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin)
 
## Moviescope Task

### Transfer Learning
In this part, the fine-tuned model weights are being frozen. The Moviescope training dataset is fed to the model and the visual and textual representations are written to `--rep_save_path` so they can be used later to train a regressor. For this you need to have prepared the captions (see below, captions_preparation.py) and extracted visual features as explained below. The path to captions is not passed as an argument here but is created in /datasets/me_dataset.py ( combination of dataroot in yaml file and hard coded things). If file does not exist, another task is called, so be careful with this. Todo = Change the code here adding more complete error messages

```
python script/ME/vilbert_representations.py --bert_model bert-base-uncased --from_pretrained <path_to_the_pretrained_model> --config_file config/bert_base_6layer_6conect.json --tasks 19 --split trainval --batch_size 128 --task_specific_tokens --rep_save_path datasets/ME/out_feaures/train_features.pkl
```
To avoid the script to evaluate every time, any name can be specified in the type of task in the yaml file. It will then return loss = 0 and score = 0, but the features will be extracted.

### Training and evaluation
The training and evaluation of the classifier on the frozen data are made by:
```
python script/train_classifier
```
This uses a 4 layers (size 512, 64, 32, 13) neural network with multi label margin loss which outputs the probability of being of any genre, a threshold is then set to 0.5 to tell wether the film is or not of 1 genre. The loss curve is saved in "/data/moviescope/outputs/train_loss.png" (hardcoded) and returns the mean Average Precision (mAP). The mAP is computed by comparing every genre of every output with the ground truth.


## License

vilbert-multi-task is licensed under MIT license available in [LICENSE](LICENSE) file.
