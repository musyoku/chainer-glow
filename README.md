# Glow: Generative Flow with Invertible 1×1 Convolutions

[https://arxiv.org/abs/1807.03039](https://arxiv.org/abs/1807.03039)

後で実装解説を書きます。

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

[Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

- [celeba-32x32-images-npy.zip](https://drive.google.com/open?id=1HnaTektDZGwyjRwv08wBejVPsMTiSu1t)
- [celeba-64x64-images-npy.zip](https://drive.google.com/open?id=14XkuMovCGdJp2Nz6RLs85irM0_a7PKnE)
- [celeba-128x128-images-npy.zip](https://drive.google.com/open?id=197IFPFaj-HS0KEOZS56ycQP-Sz3b3_m1)


# Run

```
cd run
python3 train.py -dataset /home/user/dataset/celeba-64x64-images-npy/ -b 4 -depth 32 -levels 4 -nn 512 -bits 5 -ext npy
```

# Results

## Effect of change of #channels

```
cd run/experiments
python3 change_channels.py -snapshot-1 ../snapshot_512 -snapshot-2 ../snapshot_256 -snapshot-3 ../snapshot_128
```

![https://thumbs.gfycat.com/UnimportantYoungEider-size_restricted.gif](https://thumbs.gfycat.com/UnimportantYoungEider-size_restricted.gif)

## Effect of change of temperature

```
cd run/experiments
python3 change_temperature.py -snapshot ../snapshot
```

![https://thumbs.gfycat.com/WeeWelltodoGrasshopper-size_restricted.gif](https://thumbs.gfycat.com/WeeWelltodoGrasshopper-size_restricted.gif)

![https://thumbs.gfycat.com/LinedAptInexpectatumpleco-size_restricted.gif](https://thumbs.gfycat.com/LinedAptInexpectatumpleco-size_restricted.gif)

## Interpolation

```
cd run/experiments
python3 interpolation.py -snapshot ../snapshot -dataset /home/user/dataset/celeba-64x64-images-npy/ -ext npy -temp 1
```

![https://thumbs.gfycat.com/TautLegalJellyfish-size_restricted.gif](https://thumbs.gfycat.com/TautLegalJellyfish-size_restricted.gif)

![https://thumbs.gfycat.com/PoshMatureEagle-size_restricted.gif](https://thumbs.gfycat.com/PoshMatureEagle-size_restricted.gif)

## CelebA HQ

```
cd run/experiments
python3 generate.py  -snapshot ../snapshot_hq --temperature 0.7
```

8 GPUs with 16GB memory were used in this experiment and it took 4 days to get the result.

![https://thumbs.gfycat.com/ConsciousAlarmingAmericanredsquirrel-size_restricted.gif](https://thumbs.gfycat.com/ConsciousAlarmingAmericanredsquirrel-size_restricted.gif)

```
cd run/experiments
python3 random_walk.py  -snapshot ../snapshot_hq --temperature 0.7
```

![https://thumbs.gfycat.com/FaintWideeyedAfricanporcupine-size_restricted.gif](https://thumbs.gfycat.com/FaintWideeyedAfricanporcupine-size_restricted.gif)