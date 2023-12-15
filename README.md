
# Official codebase for _BridgeQA: Bridging the Gap between 2D and 3D Visual Question Answering: A Fusion Approach for 3D VQA_.

## Acknowledgements
We would like to thank [facebookresearch/votenet](https://github.com/facebookresearch/votenet) for the 3D object detection, [daveredrum/ScanRefer](https://github.com/daveredrum/ScanRefer) for the 3D localization codebase and [ScanQA](https://github.com/ATR-DBI/ScanQA/) for 3D question answering codebase.
We also thank [BLIP](https://github.com/salesforce/BLIP/) for the 2D Vision-Language Model and architecture codebase.
<!-- [facebookresearch/votenet](https://github.com/daveredrum/ScanRefer) for the 3D object detection codebase and [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) for the CUDA accelerated PointNet++ implementation. -->

## TODO
- [x] Make copy of BLIP codes
- [x] Clean-up model codes
- [x] Clean-up training codes
- [x] Test training
- [ ] Clean-up prediction codes
- [ ] Test prediction
- [ ] Clean-up detector pre-training.
- [ ] Test detector pre-training.
- [x] Clean-up dependencies.
- [ ] Report performance with this cleaned implementation
- [ ] Add and combine SQA3D training codes

## Results
### ScanQA
|        Method       |     EM@1    |     B-1     |     B-4     |      R      |      M      |      C      |
|:-------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|       Reported      | 31.29/30.82 | 34.49/34.41 | 24.06/17.74 | 43.26/41.18 | 16.51/15.60 | 83.75/79.34 |
| This Implementation |             |             |             |             |             |             |
### SQA
|        Method       |  Acc  |
|:-------------------:|:-----:|
|       Reported      | 52.91 |
| This Implementation |       |

## License
BridgeQA is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License](LICENSE).

Copyright (c) 2023.
