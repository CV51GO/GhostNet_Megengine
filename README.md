## 介绍

本项目是论文《GhostNet: More Features from Cheap Operations》的Megengine实现。该论文的官方实现地址：https://github.com/huawei-noah/ghostnet


## 环境安装

依赖于CUDA10

```
conda create -n GhostNet python=3.7
pip install -r requirements.txt
```

## 使用方法

安装完环境后，直接运行`python compare.py`。

`compare.py`文件对官方实现和Megengine实现的推理结果进行了对比。

运行`compare.py`时，会读取`./data`中存放的图片进行推理。`compare.py`中实现了Megengine框架和官方使用的Pytorch框架的推理，并判断两者推理结果的一致性。