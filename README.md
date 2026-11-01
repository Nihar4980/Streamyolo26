# Streamyolo26

## streaming perception object detection for Autonmous Vehicle
<p align='left'>
  <img src='https://github.com/Nihar4980/Streamyolo26/blob/main/assests/Streamyolo26_flowchart.png' width='721'/>
</p>


## Benchmark

|Model |size |velocity | sAP<br>0.5:0.95 | sAP50 |sAP75| weights | COCO pretrained weights |
| ------        |:---: | :---:       |:---:     |:---:  | :---: | :----: | :----: |
|Streamyolo26-n    |600×960  |1x      |24.4     |42.1 | 24.2 |[github](https://github.com/Nihar4980/Streamyolo26/blob/main/releases/download/Streamyolo26n.pt) |[github](https://github.com/Nihar4980/Streamyolo26/blob/main/releases/download/yolo26n.pt) |
|Streamyolo26-s    |600×960  |1x      |33.8     |53.6 | 34.9 |[github](https://github.com/Nihar4980/Streamyolo26/blob/main/releases/download/Streamyolo26s.pt) |[github](https://github.com/Nihar4980/Streamyolo26/blob/main/releases/download/yolo26s.pt) |
|Streamyolo26-m    |600×960  |1x      |37.2     |55.8 | 39.8 |[github](https://github.com/Nihar4980/Streamyolo26/blob/main/releases/download/Streamyolo26m.pt) |[github](https://github.com/Nihar4980/Streamyolo26/blob/main/releases/download/yolo26m.pt) |
|Streamyolo26-l   |600×960  |1x      |40.0     |59.2 | 41.9 |[github](https://github.com/Nihar4980/Streamyolo26/blob/main/releases/download/Streamyolo26l.pt) |[github](https://github.com/Nihar4980/Streamyolo26/blob/main/releases/download/yolo26l.pt) |
|Streamyolo26-x    |600×960  |1x      |42.0     |62.9 | 43.8 |[github](https://github.com/Nihar4980/Streamyolo26/blob/main/releases/download/Streamyolo26x.pt) |[github](https://github.com/Nihar4980/Streamyolo26/blob/main/releases/download/yolo26x.pt) |


## Quick Start

<details>
<summary>Dataset preparation</summary>

You can download Argoverse-1.1 full dataset and annotation from [HERE](https://www.cs.cmu.edu/~mengtial/proj/streaming/) and unzip it.

The folder structure should be organized as follows before our processing.

```shell
ISYOLO
├── exps
├── tools
├── yolox
├── data
│   ├── Argoverse-1.1
│   │   ├── annotations
│   │       ├── tracking
│   │           ├── train
│   │           ├── val
│   │           ├── test
│   ├── Argoverse-HD
│   │   ├── annotations
│   │       ├── test-meta.json
│   │       ├── train.json
│   │       ├── val.json
```

The hash strings represent different video sequences in Argoverse, and `ring_front_center` is one of the sensors for that sequence. Argoverse-HD annotations correspond to images from this sensor. Information from other sensors (other ring cameras or LiDAR) is not used, but our framework can be also extended to these modalities or to a multi-modality setting.


</details>


<details>
<summary>Installation</summary>

```shell
# basic python libraries    py3.7+ torch1.7.1+mmcv1.1.5   or  py3.9+torch1.13.1+mmcv2.0.0
conda create --name streamyolo python=3.7         # or conda create --name streamyolo python=3.9  

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# or torch1.13.1+cu116   

pip3 install yolox==0.3
git clone git@github.com:GjtZ/ISYOLO.git



cd ISYOLO/

# add ISYOLO to PYTHONPATH and add this line to ~/.bashrc or ~/.zshrc (change the file accordingly)
ADDPATH=$(pwd)
echo export PYTHONPATH=$PYTHONPATH:$ADDPATH >> ~/.bashrc
source ~/.bashrc

# Installing `mmcv` for the official sAP evaluation:
# Please replace `{cu_version}` and ``{torch_version}`` with the versions you are currently using.
# You will get import or runtime errors if the versions are incorrect.
pip install mmcv-full==1.1.5 -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
# pip install mmcv-full==1.1.5 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
# pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html     
```

</details>


<details>
<summary>Reproduce our results on Argoverse-HD</summary>

Step1. Prepare COCO dataset
```shell
cd <ISYOLO_HOME>
ln -s /path/to/your/Argoverse-1.1 ./data/Argoverse-1.1
ln -s /path/to/your/Argoverse-HD ./data/Argoverse-HD
```

Step2. Reproduce our results on Argoverse:

```shell
python tools/train.py -f cfgs/m_s50_onex_dfp_tal_flip.py -d 8 -b 32 -c [/path/to/your/coco_pretrained_path] -o --fp16
```
* -d: number of gpu devices.
* -b: total batch size, the recommended number for -b is num-gpu * 8.
* --fp16: mixed precision training.
* -c: model checkpoint path.

</details>


<details>
<summary>Offline Evaluation</summary>

We support batch testing for fast evaluation:

```shell
python tools/eval.py -f  cfgs/l_s50_onex_dfp_tal_flip.py -c [/path/to/your/model_path] -b 64 -d 8 --conf 0.01 [--fp16] [--fuse]
```
* --fuse: fuse conv and bn.
* -d: number of GPUs used for evaluation. DEFAULT: All GPUs available will be used.
* -b: total batch size across on all GPUs.
* -c: model checkpoint path.
* --conf: NMS threshold. If using 0.001, the performance will further improve by 0.2~0.3 sAP.

</details>

<details>
<summary>Online Evaluation</summary>

We modify the online evaluation from [sAP](https://github.com/mtli/sAP)

Please use 1 V100 GPU to test the performance since other GPUs with low computing power will trigger non-real-time results!!!!!!!!

```shell
cd sAP/streamyolo
bash my_streamyolo.sh
```

</details>



## Citation


## License
This repo is released under the Apache 2.0 license. Please see the LICENSE file for more information.

