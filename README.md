# VGPNet

This is the official implementation of the paper [VGPNet: a Vision-aided GNSS Positioning Framework with Cross-Channel Feature Fusion for Urban Canyons]().

## News

- 🎉 (2025.09) We have released the code and dataset for VGPNet. Please refer to the instructions below to set up the environment and run the code.

## Requirements
- Python 3.10
- PyTorch 2.7.1

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/hu-xue/VGPNet
   cd VGPNet
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained models and datasets as specified in the paper.

    The processed GNSS and fisheye image datasets can be found at: [BaiduNet](https://pan.baidu.com/s/1tOnJjV6vcdCXzLw6WfNHgQ?pwd=cywn)
    
    Download the processed datasets and place them in the `data/` directory. See the structure below:
    ```
    VGPNet/
    ├── data/
    │   ├── hksc-night/
    │   │   ├── hksc252l.25o
    │   │   └── ...
    │   ├── night_image/
    │   │   ├── 1757416968.022379.jpg
    │   │   └── ...
    │   ├── night.obs
    │   ├── night_fix.csv
    │   └── ...
    ├── config/
    ├── train.py
    ├── predict.py
    └── ...
    ```

4. Run the training and evaluation scripts as described in the paper.
    ```bash
    # Example command to train the model
    python train.py --config_file config/rw1_train.json
    
    # Example command to evaluate the model
    python predict.py --config_file config/rw2_predict.json
    ```

## Thanks
We would like to thank the authors of the following repositories for their open-source contributions, which have been helpful for our work:
- [TDL-GNSS](https://github.com/ebhrz/TDL-GNSS)
- [pyrtklib](https://github.com/IPNL-POLYU/pyrtklib)


