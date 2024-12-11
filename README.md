# CASPFuse
 CASPFuse: An Infrared and Visible Image Fusion Method based on Dual-cycle Crosswise Awareness and Global Structure-tensor Preservation
![alt text](img/framework.png)

## Datasets
The training and testing data of our CASPFuse can be downloaded from: [RoadScene](https://github.com/hanna-xu/RoadScene), [M3FD](https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6) and [PET-MRI](https://www.med.harvard.edu/AANLIB/home.html)

## Get start
```python
## training
   python train.py 
```

```python
## testing
   python test.py 
```

## Citation
```
@InProceedings{Li_2024_CVPR,
    author    = {Li, Xuan and Chen, Rongfu and Wang, Jie and Ma, Lei and Cheng, Li and Yuan, Haiwen},
    title     = {DSTCFuse: A Method based on Dual-cycled Cross-awareness of Structure Tensor for Semantic Segmentation via Infrared and Visible Image Fusion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {5558-5567}
}

@ARTICLE{10776019,
  author={Li, Xuan and Chen, Rongfu and Wang, Jie and Chen, Weiwei and Zhou, Huabing and Ma, Jiayi},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={CASPFuse: An Infrared and Visible Image Fusion Method Based on Dual-Cycle Crosswise Awareness and Global Structure-Tensor Preservation}, 
  year={2025},
  volume={74},
  number={},
  pages={1-15},
  keywords={Image fusion;Image edge detection;Transformers;Generators;Low-pass filters;Feature extraction;Semantics;Interference;Information filters;Image reconstruction;Deep learning;image fusion;modality transition;structure awareness},
  doi={10.1109/TIM.2024.3509580}}
```
