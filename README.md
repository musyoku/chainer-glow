# :construction: Work in Progress :construction:

# Glow: Generative Flow with Invertible 1×1 Convolutions

[https://arxiv.org/abs/1807.03039](https://arxiv.org/abs/1807.03039)

# Requirements

- Python 3
- Ubuntu
- Chainer 4
- Extremely huge amount of GPU memory
- python-tabulate
- scipy

# Installation

```
pip3 install chainer cupy h5py
pip3 install tabulate
```

# Dataset
## CelebA HQ

- [celeba-32x32-images-npy.zip](https://drive.google.com/open?id=1HnaTektDZGwyjRwv08wBejVPsMTiSu1t)
- [celeba-64x64-images-npy.zip](https://drive.google.com/open?id=14XkuMovCGdJp2Nz6RLs85irM0_a7PKnE)
- [celeba-128x128-images-npy.zip](https://drive.google.com/open?id=197IFPFaj-HS0KEOZS56ycQP-Sz3b3_m1)


## Danbooru2017

Coming soon

# Results

```
python3 run/train.py -dataset /home/user/dataset/celeba-64x64-images-npy/ -b 4 -depth 32 -levels 4 -nn 512 -bits 5 -ext npy -learn-z
```

## Effect of change of #channels

![https://thumbs.gfycat.com/WellmadeBlankJaguar-size_restricted.gif](https://thumbs.gfycat.com/WellmadeBlankJaguar-size_restricted.gif)

## Effect of change of temperature

![https://thumbs.gfycat.com/WeeWelltodoGrasshopper-size_restricted.gif](https://thumbs.gfycat.com/WeeWelltodoGrasshopper-size_restricted.gif)