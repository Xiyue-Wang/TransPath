
## TransPath(coming soon)

The new better and stronger pre-trained models have [released](https://github.com/Xiyue-Wang/RetCCL)
#### Hardware

* 128GB of RAM
* 32*Nvidia V100 32G GPUs

### Preparations
1.Download all [TCGA](https://portal.gdc.cancer.gov/projects?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22projects.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%5D%7D) WSIs.

2.Download all [PAIP](http://wisepaip.org/paip) WSI

We crop these WSIs into patch images.we randomly select 100 images from each WSI.Finally,So, there will be about 2,700,521 unlabeled histopathological
images.If you want these images, you can contact me.

### Usage: Pre-Training Vision Transformers for histopathology images

#### 1.MoCo v3 
We also trained [MoCo v3](https://arxiv.org/abs/2104.02057) on these histopathological images.
The pre-trained  models can be downloaded as following:

[vit_small]()

[vit_conv_small]()

##### Usage: Self-supervised Pre-Training
please see the [instructions](https://github.com/facebookresearch/moco-v3)

##### Usage: Get frozen features

```
python get_features.py \
        -a vit_small
```
##### Usage: End-to-End Fine-tuning ViT
To perform end-to-end fine-tuning for ViT, use our script to convert the pre-trained ViT checkpoint to [DEiT](https://github.com/facebookresearch/deit) format:
```
python convert_to_deit.py \
  --input [your checkpoint path]/[your checkpoint file].pth.tar \
  --output [target checkpoint file].pth
```
Then run the training (in the DeiT repo) with the converted checkpoint:
```
python $DEIT_DIR/main.py \
  --resume [target checkpoint file].pth \
  --epochs 150
```




