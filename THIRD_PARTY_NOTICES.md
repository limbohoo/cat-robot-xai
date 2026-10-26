# Third-party data, models and licences

This repository contains no images from the datasets below, no test recordings and no frames of third-party videos. It does contain model weights trained on these datasets.

| Item | Used for | Licence / terms | What it means here |
| --- | --- | --- | --- |
| CatFLW (Martvel et al., 2023) | Training `face.pt` | CC BY-NC 4.0 | `face.pt` and `face.onnx` are for non-commercial use, with attribution |
| Animal Pose dataset (Cao et al., ICCV 2019) | Training `body.pt` | Terms of the original repository | Cite the paper when you use `body.pt` |
| PASCAL VOC 2012 images | Images for the Animal Pose annotations | VOC terms; images come from Flickr and keep their own copyright | Images are not redistributed |
| COCO | Pretraining of the YOLOv8 detector | CC BY 4.0 (annotations) | Used only through the pretrained detector |
| YOLOv8 (Ultralytics) | Cat detector | AGPL-3.0 | Weights are downloaded by Ultralytics, not stored here. A product that ships the detector must follow AGPL-3.0 or use an Ultralytics licence |
| MobileNetV3 ImageNet weights (torchvision) | Backbone initialisation | BSD-3-Clause (torchvision) | Starting point for our keypoint networks |
| PyTorch, torchvision, OpenCV, NumPy, SciPy, FastAPI | Runtime | BSD / Apache-2.0 / MIT | No extra conditions |

The Colab tutorial and the evaluation logs mention a 35-second web video ("My cat thinks she's home alone with a ghost – Part 2", channel Kiki The Dream Cat, YouTube). Only numbers and keypoints derived from it are reported; the video and its frames are not included.

## Citations

- Martvel, G., Farhat, N., Shimshoni, I., & Zamansky, A. (2023). CatFLW: Cat Facial Landmarks in the Wild Dataset. arXiv:2305.04232.
- Martvel, G., Shimshoni, I., & Zamansky, A. (2024). Automated detection of cat facial landmarks. International Journal of Computer Vision. https://doi.org/10.1007/s11263-024-02006-w
- Cao, J., Tang, H., Fang, H.-S., Shen, X., Lu, C., & Tai, Y.-W. (2019). Cross-domain adaptation for animal pose estimation. ICCV.
- Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The PASCAL Visual Object Classes (VOC) Challenge. International Journal of Computer Vision.
- Howard, A., et al. (2019). Searching for MobileNetV3. ICCV.
- Jocher, G., Chaurasia, A., & Qiu, J. Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
