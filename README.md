# PairDETR: face_body_detection_and_association
This repository contains the official implementation of PairDETR, a method for Joint Detection and Association of Human Bodies and Faces.
<img src="./assest/teaser.jpg"></img>
## Getting started
### Installation:
For setting up the environment, we highly recommend using Docker images to ensure reproducibility and avoid any dependency issues. For our experiments, we used the Docker image 
```
skypirate91/minigpt4:0.4
```
You can also use the provided requirements file to set up your personal environment.
```
pip install -r requirements.txt
```
### Datasets
#### CrowdHuman
* Download the dataset from <a  href="https://www.crowdhuman.org/download.html">here</a>.
* We used the annotations prepared by the authors of <a  href="https://openaccess.thecvf.com/content/ICCV2021/html/Wan_Body-Face_Joint_Detection_via_Embedding_and_Head_Hook_ICCV_2021_paper.html">BFJDet</a>  <a  href="https://drive.google.com/drive/folders/12ypJ8gB7v4T1_blYraGslCRK9hXuNGBP">download</a>.
* Preprocessed annotations that cut boxes outside the image frame and removed ignored boxes are available in the annotations folder.
#### CityPersons
* Download the dataset from <a  href="https://www.cityscapes-dataset.com/">here</a>.
* We used the annotations prepared by the authors of <a  href="https://openaccess.thecvf.com/content/ICCV2021/html/Wan_Body-Face_Joint_Detection_via_Embedding_and_Head_Hook_ICCV_2021_paper.html">BFJDet</a>  <a  href="https://drive.google.com/drive/folders/1Sk2IAmm_wTVh289RKs5FiU17siWrJJCu">download</a>.
* We preprocessed the annotations to cut the boxes located outside the image frame and removed ignore boxes, you can use preprocessed ones directly from annotations folder.
## Training
After setting up the environment and preparing the datasets, update the paths in <a href='./config.py'>config.py</a>.

Then to start the training run:
```
python run_lightning_d.py
```
After that the training should start.
You can experiment with different hyperparameters and training setups.
We used huggingface model loader to make it easier to  experiment with other backbones or model, refer to <a href='./train_lightning_d.py'>train_lightning_d.py</a> Detr_light class initialization you can change the model for example to:
```
DetaForObjectDetection.from_pretrained(
            "jozhang97/deta-swin-large", 
            two_stage = False, with_box_refine = False
    )
``` 
Please keep in mind that we don't support two-stages or box refinement training yet.
you can also experiment with different feature extractors using timm integration. Keep in mind that these backbones are trained on ImageNet, not COCO so you may consider increasing the number of epochs for training and reseting some hyperparameters:
```
DeformableDetrForObjectDetection.from_pretrained(
            "SenseTime/deformable-detr",
            use_timm_backbone=True, 
            backbone="mobilenetv3_small_050.lamb_in1k"
    )
```
## Inference
refer to our test.py script for loading the model and inference, simple example to load the model:
```
from train_lightning_d import Detr_light
model = Detr_light(num_classes = 3, num_queries = 1500)
checkpoint = torch.load(<path to the chk>, map_location="cuda")
model.load_state_dict(checkpoint, strict=False)
```
<a href=''>![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)</a>
## Results
Comparison between PairDETR method and other methods in the miss matching rate mMr-2 (the lower the better):

| **Model** | **Reasnable** | **Bare** | **Partial** | **Heavy** | **Hard** | **Average** |**Checkpoints** |
|-----------|:-------------:|:--------:|-------------|:---------:|----------|----------|----------|
| **POS**   |     55.49     |   48.20  | 62.00       |   80.98   | 84.58    |   66.4  | <a href="https://drive.google.com/file/d/1GFnIXqc9aG0eXSQFI4Pe4XfO-8hAZmKV/view">weights</a> |
| **BFJ**   |     42.96     |   37.96  | 48.20       |   67.31   | 71.44    |   52.5  | <a href="https://drive.google.com/file/d/1E8MQf3pfOyjbVvxZeBLdYBFUiJA6bdgr/view">weights</a> |
| **BPJ**   |     -     |   -  | -      |   -   | -    |   50.1  |<a href="https://github.com/hnuzhy/BPJDet">weights</a> |
| **PBADET**   |     -     |   -  | -      |   -   | -    |   50.8  | <a href="">none</a> |
| **OURs**  |     35.25     |   30.38  | 38.12       |   52.47   | 55.75    |   42.9  | <a href="">weights</a> |
## Useful links
### Papers
* <a href='https://arxiv.org/abs/2005.12872'>End-to-End Object Detection with Transformers</a>
* <a href='https://arxiv.org/abs/1805.00123'>CrowdHuman: A Benchmark for Detecting Human in a Crowd</a>
* <a href='https://openaccess.thecvf.com/content/ICCV2021/html/Wan_Body-Face_Joint_Detection_via_Embedding_and_Head_Hook_ICCV_2021_paper.html'>Body-Face Joint Detection via Embedding and Head Hook</a>
* <a href='https://arxiv.org/abs/2010.04159'>Deformable DETR: Deformable Transformers for End-to-End Object Detection</a>
* <a href='https://arxiv.org/abs/2012.06785'>DETR for Crowd Pedestrian Detection</a>
* <a href='https://arxiv.org/abs/2204.07962'>An Extendable, Efficient and Effective Transformer-based Object Detector</a>

### This work is implemented on top of:
* <a href='https://github.com/facebookresearch/detr/tree/3af9fa878e73b6894ce3596450a8d9b89d918ca9'>DETR</a>
* <a href='https://github.com/fundamentalvision/Deformable-DETR'>Deformable-DETR</a>
* <a href='https://github.com/AibeeDetect/BFJDet/tree/main'>BFJDet</a>
* <a href='https://huggingface.co/docs/transformers/en/index'>Hugginface transformers</a>
