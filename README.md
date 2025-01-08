# FreeFusion



### 1. Recommended Environment
- Python 3.7
- Pytorch 1.9.1+cu111
- scipy 1.7.3
- numpy 1.21.6
- opencv-python 4.5.2

### 2. Data Preparation

Thanks to the Potsdam, WHU, MFNet, LLVIP and M3FD datasets contributors. To generate the experiment results as described in our paper, you can download Potsdam, WHU, MFNet, LLVIP and M3FD datasets at [[Baidu Yun](https://pan.baidu.com/s/117ckQSjeN5UQ5qUDSQ2PfQ?pwd=IVIF)].  **Note that**, if you want to train and test the Potsdam, WHU and MFNet datasets, you need to run `crop.py` to crop them. LLVIP and M3FD datasets are tested directly, without cropping.

### 3. Training

If you want to train our FreeFusion, You should list your dataset as followed rule:

```python
-- dataset
    -- train
        -- your_dataset
             -- ir
            	 -- input
                	|-- xxxx.png
                    |-- ......
            	 -- target
               		|-- xxxx.png
                    |-- ......
             -- rgb
            	 -- input
                	|-- xxxx.png
                    |-- ......
            	 -- target
               		|-- xxxx.png
                    |-- ......
             --	seg
            	|-- xxxx.png
                |-- ......
```

Then, please run the following prompt:

```python
python train.py
```

Finally, the trained model is available in `'./checkpoints/your_dataset/'`. Training information (batch, epoch, etc.) can be changed in the `'training.yml'`.

### 4. Testing

- **Pretrained models**

Pretrained models is available in `'. /checkpoints/potsdam/model_potsdam.pth'`, `'. /checkpoints/whu/model_whu.pth'` and `'. /checkpoints/mfnet/model_mfnet.pth'`, which are responsible for Potsdam, WHU, and MFNet, respectively.

- **Results in our paper**

If you want to infer our FreeFusion and get the fusion results on Potsdam, WHU and MFNet datasets, please place the paired images into `'./dataset/test/potsdam'`, `'./dataset/test/whu'` and `'./dataset/test/mfnet'`.

Then, please run the following prompt:

```python
python test_potsdam.py
```

```python
python test_whu.py
```

```python
python test_mfnet.py
```

If you want to obtain the fusion results for the generalizability experiment on LLVIP and M3FD datasets, please place the paired images into `'./dataset/test/LLVIP'` and `'./dataset/test/M3FD'`. **Notice that,** we select the fusion model trained on WHU dataset for testing them.

Then, please run the following prompt:

```python
python test_LLVIP.py
```

```python
python test_M3FD.py
```
