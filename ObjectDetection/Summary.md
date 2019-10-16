# 本部分根据如下路线图小结了几篇重要论文

![Route](./Route.png)

## Reference
[AwesomeObjectDetection](https://github.com/amusi/awesome-object-detection)

## Summary

|名称|年份|会议/期刊/杂志|地址|总结
|:---:|:---:|:---:|:---:|:---:|
|R-CNN|2013|LSVRC|[arXiv](http://arxiv.org/abs/1311.2524)|[R-CNN](#R-CNN)|
|Fast R-CNN|2015|ICCV|[arXiv](http://arxiv.org/abs/1504.08083)|[Fast R-CNN](#Fast&nbsp;R-CNN)|
|Faster R-CNN|2015|NIPS|[arXiv](http://arxiv.org/abs/1506.01497)|[Faster R-CNN](#Faster%20R-CNN)|
|YOLO v1|2016|CVPR|[arXiv](http://arxiv.org/abs/1506.02640)|[YOLO v1](#YOLO%20v1)|
|SSD|2016|ECCV|[arXiv](http://arxiv.org/abs/1512.02325)|[SSD](#SSD)|
|R-FCN|2016|NIPS|[arXiv](http://arxiv.org/abs/1605.06409)|[R-FCN](#R-FCN)|
|YOLO v2|2017|CVPR|[arXiv](https://arxiv.org/abs/1612.08242)|[YOLO v2](#YOLO%20v2)|
|FPN|2017|CVPR|[arXiv](https://arxiv.org/abs/1612.03144)|[FPN](#FPN)|
|RetinaNet|2017|ICCV|[arXiv](https://arxiv.org/abs/1708.02002)|[RetinaNet](#RetinaNet)|
|Mask R-CNN|2017|ICCV|[arXiv](http://arxiv.org/abs/1703.06870)|[Mask R-CNN](#Mask%20R-CNN)|
|YOLO v3|2018|arXiv|[arXiv](https://arxiv.org/abs/1804.02767)|[YOLO v3](#YOLO%20v3)|
|RefineNet|2018|CVPR|[arXiv](https://arxiv.org/abs/1711.06897)|[RefineNet](#RefineNet)|
|CornerNet|2018|ECCV|[arXiv](https://arxiv.org/abs/1808.01244)|[CornerNet](#CornerNet)|
|M2Det|2019|AAAI|[arXiv](https://arxiv.org/abs/1811.04533)|[M2Det](#M2Det)|


## R-CNN
* Two-stage，首次提出用深度学习的方法来进行目标检测
* 使用Selective Search的方法从图片上选取了2K+个候选区域，并送入CNN中获取固定长度的向量
* 由于使用AlexNet进行特征向量提取（4096维），候选区域被统一变换为227*227尺寸
* 预训练使用在ILSVRC2012提取的模型和权重，然后将最后一层分类层替换为相应的输出（如VOC输出20类+1背景维）进行fine-tuning，与GT的IoU>0.5的被视为正样本，否则为负样本
* 使用特征向量对每个类都训练一个SVM，GT对应的特征向量为正样本，与GT的IoU<0.3的为负样本，其余丢弃，最后使用非极大抑制确定分类
* 作者还训练了一个线性的边框回归模型
   
## Fast R-CNN
* Two-stage，依旧使用了Selective search的方法获取候选区域，但是实现了参数共享，直接在feature map上找到各候选区域对应位置
* 提出了RoI Pooling layer（用于替代VGG第5层的普通池化层），直接将feature map均分为M*N块，得到固定大小的特征向量
   
## Faster R-CNN

   
## YOLO v1

   
## SSD

   
## R-FCN


## YOLO v2


## FPN


## RetinaNet


## Mask R-CNN


## YOLO v3


## RefineNet


## CornerNet


## M2Det


## 后续待补充
