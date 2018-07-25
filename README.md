# :construction: Work in Progress :construction:

# Glow: Generative Flow with Invertible 1×1 Convolutions

[https://arxiv.org/abs/1807.03039](https://arxiv.org/abs/1807.03039)

**Todo**

- [x] Implement inference model
- [x] Implement generative model
- [x] Implement training loop
- [ ] Quantitative experiments
- [x] LU decomposition
- [x] Debug

# Requirements

- Python 3
- Ubuntu
- Chainer 4
- Extremely huge amount of GPU memory
- python-tabulate

# Installation

```
pip3 install chainer cupy h5py
pip3 install tabulate
```

# Dataset
## CelebA HQ

[http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

[https://github.com/nperraud/download-celebA-HQ](https://github.com/nperraud/download-celebA-HQ)


## Danbooru 2017

# Results

```
python3 run/train.py -dataset /home/user/dataset/celeba-64x64-images-npy/ -b 4 -depth 32 -levels 4 -nn 512 -bits 5 -ext npy -learn-z
```

![https://thumbs.gfycat.com/WellmadeBlankJaguar-size_restricted.gif](https://thumbs.gfycat.com/WellmadeBlankJaguar-size_restricted.gif)

```
python3 run/train.py -dataset /home/user/dataset/celeba-128x128-images-npy/ -b 4 -depth 32 -levels 4 -nn 512 -bits 5 -ext npy -learn-z
```

![https://thumbs.gfycat.com/TerribleThankfulAardwolf-size_restricted.gif](https://thumbs.gfycat.com/TerribleThankfulAardwolf-size_restricted.gif)