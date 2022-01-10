Codes and model parameters for our paper in AAAI-2022, "**ACDNet: Adaptively Combined Dilated Convolution for Monocular Panorama Depth Estimation**". (https://arxiv.org/abs/2112.14440)


**Inference**

Please download the pretrained model parameters [here](https://drive.google.com/drive/folders/1f7D6b_UypKwnL6Rsxq4CQlQaLu_anXqL?usp=sharing) and put it in folder `./checkpoints`. The inference process is executed with the following commands:

```python infer.py --gpus 0 --checkpoints ./checkpoints/acdnet-m3d.pt --example ./examples/m3d.png```

**TODO**
- [x] models and parameters
- [ ] training codes