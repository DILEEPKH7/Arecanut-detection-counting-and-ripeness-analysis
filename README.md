# Arecanut-detection-counting-and-ripeness-analysis

### Dependencies
The source code in this repository is written in Python 3.8. This section gives an overview of how to set-up the source code on a computer.

### Clone this repository
Follow Github's [Git Guides](https://github.com/git-guides) to set-up Git locally on your machine. Then clone this repository to download the code.

### Conda environment
Download [Anaconda](https://www.anaconda.com) and run the following command to set-up a new virtual environment with all the necessary Python packages and libraries: 
`conda env create -f environment.yml` 
Alternatively, use the [Anaconda Navigator GUI](https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/#importing-an-environment).

To *activate* the environment, run `conda activate my_environment` before launching the Python IDE, or use the[Anaconda Navigator GUI](https://docs.anaconda.com/anaconda/navigator/).

### Data structure
- make sure your dataset structure as follows:
```
├── coco
│   ├── annotations
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── images
│   │   ├── train2017
│   │   └── val2017
│   ├── labels
│   │   ├── train2017
│   │   ├── val2017
│   ├── LICENSE
│   ├── README.txt
```
### Training
```shell 
python tools/train.py --batch 2 --conf configs/yolov6s_finetune.py --data data/areca.yaml --fuse_ab --device 0 --img-size 1024
```

### Evaluation
```shell
python tools/eval.py --data data/areca.yaml  --weights runs/train/exp257/weights/best_ckpt.pt --task val --device 0	
```

### Inference
```shell
 python infer.py --weights runs/train/exp/weights/best_ckpt.pt --source data/images/ --device 0 --yaml data/areca.yaml
```


