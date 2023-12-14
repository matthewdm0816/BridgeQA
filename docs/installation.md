## Installation

This code is based on [ScanRefer](https://github.com/daveredrum/ScanRefer) and [ScanQA](https://github.com/ATR-DBI/ScanQA/). Please also refer to the ScanRefer and ScanQA setup.

<!-- - Install PyTorch:
    ```shell
    conda install pytorch==1.12.1 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
    ``` -->
- Install PyTorch: `pytorch==1.12.1 torchvision==0.13.1` compatible your CUDA version.

- Install the necessary packages with `requirements.txt`:
    ```shell
    pip install -r requirements.txt
    ```

- Compile the CUDA modules for the PointNet++ backbone:
    ```shell
    cd lib/pointnet2
    python setup.py install
    ```

Note that this code has been tested with Python 3.9.7, pytorch 1.12.1, and CUDA 11.3 on Ubuntu 20.04.1.
