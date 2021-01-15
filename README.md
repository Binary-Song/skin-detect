# 简单皮肤检测

简单的机器学习皮肤检测算法。

## 简介

skin-detect.py中的`skin_detect`函数将图片中的肤色像素保留，其余像素涂黑。

- `input_path`,`output_path` 输入图像输出图像的路径。
- `threshold` 越大，则判定的标准越严格（应小于等于1）。
- `w`,`h` 输出图像的长宽，调低一点不然会很慢。
## 依赖

skin_dataset.txt是训练集，前三个数表示RGB分量，第四个数表示是不是皮肤色。

要运行脚本，请先
```
pip install numpy scipy opencv-python
```

## 效果

![image](https://github.com/Binary-Song/skin-detect/blob/master/people-th0.png)

![image](https://github.com/Binary-Song/skin-detect/blob/master/people-th1.png)

![image](https://github.com/Binary-Song/skin-detect/blob/master/people-th2.png)

![image](https://github.com/Binary-Song/skin-detect/blob/master/people-th3.png)

![image](https://github.com/Binary-Song/skin-detect/blob/master/people-th4.png)

![image](https://github.com/Binary-Song/skin-detect/blob/master/people-th5.png)
