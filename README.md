# VGPNet

This is the official implementation of the paper [VGPNet: a Vision-aided GNSS Positioning Framework with Cross-Channel Feature Fusion for Urban Canyons]().

## Requirements
- Python 3.6+
- PyTorch 1.7+
- torchvision 0.8+

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

The processed GNSS and fisheye image datasets can be found at: [BaiduNet](https://pan.baidu.com/s/1MynuZFrHSqqAIWCJCd6o_Q?pwd=bra3)

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


