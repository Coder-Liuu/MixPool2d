# Torch 实现混合池化层

## 前言

该idea由我的师兄的idea改进而来，纯属自己DIY玩一玩，当时自己写这个遇到了不少困难，将算法开源希望能帮住到更多的人。

距离实现idea的日子已经过去一年多了，自己最近重新翻改了一下。

**如果您也想实现一个自己的池化层，不妨先看一下最大池化层是怎么实现的，这将会为你自定义池化层提供一定的帮助。**

- [数据集下载](https://download.csdn.net/download/qq_43497702/18340833?spm=1001.2014.3001.5503)


## 算法思想 
  对于图片中平坦地区进行模糊化，对图片中的边缘地区进行尖锐化。

## 算法步骤
```
1. 计算整张图片的均值
2. 计算池化块的均值
3. 计算池化块的最大值
4. if 若池化区域的均值 > 图片的整体均值
      该池化区域进行均值池化
   else:
      该池化区域进行最大值池化
```


  
 
