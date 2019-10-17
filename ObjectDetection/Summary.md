# 本部分根据如下路线图小结了几篇重要论文

![Route](./Route.png)

## Reference
[AwesomeObjectDetection](https://github.com/amusi/awesome-object-detection)

## Summary

|名称|年份|会议/期刊/杂志|地址|总结
|:---:|:---:|:---:|:---:|:---:|
|R-CNN|2013|LSVRC|[arXiv](http://arxiv.org/abs/1311.2524)|[R-CNN](#R-CNN)|
|Fast R-CNN|2015|ICCV|[arXiv](http://arxiv.org/abs/1504.08083)|[Fast R-CNN](#Fast-R-CNN)|
|Faster R-CNN|2015|NIPS|[arXiv](http://arxiv.org/abs/1506.01497)|[Faster R-CNN](#Faster-R-CNN)|
|YOLO v1|2016|CVPR|[arXiv](http://arxiv.org/abs/1506.02640)|[YOLO v1](#YOLO-v1)|
|SSD|2016|ECCV|[arXiv](http://arxiv.org/abs/1512.02325)|[SSD](#SSD)|
|R-FCN|2016|NIPS|[arXiv](http://arxiv.org/abs/1605.06409)|[R-FCN](#R-FCN)|
|YOLO v2|2017|CVPR|[arXiv](https://arxiv.org/abs/1612.08242)|[YOLO v2](#YOLO-v2)|
|FPN|2017|CVPR|[arXiv](https://arxiv.org/abs/1612.03144)|[FPN](#FPN)|
|RetinaNet|2017|ICCV|[arXiv](https://arxiv.org/abs/1708.02002)|[RetinaNet](#RetinaNet)|
|Mask R-CNN|2017|ICCV|[arXiv](http://arxiv.org/abs/1703.06870)|[Mask R-CNN](#Mask-R-CNN)|
|YOLO v3|2018|arXiv|[arXiv](https://arxiv.org/abs/1804.02767)|[YOLO v3](#YOLO-v3)|
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
* 使用多任务全连接层，最终分类层输出（N+1）类，包括N类目标+背景类；回归层输出4N类，表示分别为N类目标时的平移缩放
* 损失函数包括分类损失和边框回归损失，当分类为背景类时，不计算回归损失
* fine-tuning时使用正负样本1：3，与GT的IoU≥0.5的为正，[0.1,0.5)的为负样本
* 使用SVD全连接层加速网络
   
## Faster R-CNN
* Two-stage，提出了RPN（由全卷积实现）用来取代Selective Search生成候选区域,接受任意尺寸的输入，输出一系列候选区域，并且与检测网络共享参数
* 使用ZF net或者VGG net作为基础网络，得到feature map并采用3*3的sliding window处理成256维/512维向量
* 对sliding window的中心点预测anchor boxes（文中k=9，3个尺寸3种宽高比），并将256维特征映射为2k个分类层输出（foreground & background），以及4k个回归层输出（非极大抑制筛选后作为RPN输出）
* 训练RPN时，anchor box与任意目标的GT的IoU最大 或 IoU＞0.7，则被视为正样本；IoU＜0.3为负样本，其余不参与训练，取样时每个图像随机采样256个anchors来训练，并保持正负1：1，不足则用负样本填充
* 检测网络使用了Fast R-CNN，并与RPN实现了参数共享（交替训练法）
   
## YOLO v1
* 不同于生成region proposal的方法，YOLO使用整张图作为输入信息，直接在输出层回归bounding box的位置和类别，把背景误认为目标的错误更少
* 直接将一幅图分成S\*S（7*7）个网格，如果某个目标的中心位于这个网格中，则该网格负责预测该目标
* S\*S个网格中 每个网格预测B个bouding box和C个类的信息，其中每个bounding box包括4个位置值和1个confidence值（目标存在的置信度和交并比的乘积）
    输出形式为S\*S*(5\*B+C)(7\*7*30)
* sum squred error loss稍作改进作为损失函数，对小目标、紧靠的目标检测效果不佳，一个网格只有预测了2个框，损失函数有待加强
   
## SSD
* SSD采用了多尺度特征提取的思想，不断缩小特征图的长宽，越小的特征图感受野越大，以加强对各种大小的物体的检测
* 使用default box，即anchor boxes，文中采用了不同尺寸和宽高比的default box，并预测其相对于ground truth的偏移，并采取hard negative mining，
    对于m\*n的特征图，需要输出m\*n*k(c+4)维度的特征，其中k为default box数，c为分类数，文中c已经包括背景类，取背景置信度最低的负样本参与训练
    保持比例正：负=1：3
* 损失函数为smooth L1的location loss + 正负样本的softmax loss
   
## R-FCN
* 提出了位置敏感得分图（position-sensitive score map）
* 对于一个RoI，如果其中存在目标，则将此RoI划分为k\*k个区域，R-FCN在共享卷积层的最后加上了一层卷积层，输出维度为H\W\k²(C+1)，
    意思是此RoI对于每个类别，都有k²个得分图，分别对应之前RoI被划分的k\*k个区域，之后对k\*k个区域的每个区域，
    在其对应的得分图上的相应位置（RoI的左上区域，对第1个得分图的左上区域的所有值平均池化）进行池化操作，得到k²个值，分别对C+1个类进行相同操作，
    最后得到k\*k*(C+1)个值，对于每个类，取计算好的k*k个得分之和作为总得分，共C+1个总得分并使用softmax函数
* 同理从共享卷积层最后接上一个并行的得分图，用于进行位置回归，输出维度为H\*W*4k²

## YOLO v2
* 每个卷积层后均加入Batch Normalization，并移除了防止过拟合的dropout；去除了v1中的FC层，并使用了anchor boxes提高召回率
* 在v1中，YOLO先使用224\*224分类数据集训练特征提取网络，然后将输入增大到448\*448，进而使用检测数据集进行fine-tuning，v2中，在224\*224训练过的模型后，
    加入了448\*448的分类数据集的10个epochs的fine-tuning，再使用448*448的检测数据集进行fine-tuning
* 使用K-means cluster来选取anchor boxes，在检测训练集上对所有目标框进行k-means聚类最终选取5个anchor boxes
* 新的encode/decode机制，passthrough层检测细粒度特征，多尺度训练，WordTree

## FPN
* 自下而上的卷积神经网络（下采样），自上而下的过程（上采样），和两部分之间的连接
* 由于多尺度特征的融合，其对小物体检测精度有大幅度提升，为后续方案广泛借鉴

## RetinaNet
* 为了解决样本不平衡的问题，提出了Focal loss，使得数量多的样本对模型有更小的影响，模型更关注数量少的样本
* 使用Resnet+FPN+2个FCP子网络，一个负责分类，输出W\*H\*KA，；另一个负责边框回归，输出W\*H*4A，其中K为类别数目，A为anchor个数，文中A=9

## Mask R-CNN
* 保持了two-stage结构，只是另外加入了FCN来产生Mask分支（不同于其他实例分割的先分割后分类）
* 提出RoIAlign取代RoIPooling，双线性插值法避免了后者中两次引入的量化误差值（图像坐标->feature map坐标；feature map坐标->RoI feature坐标因不整除均产生误差）
* 损失函数由3个分支组成：其中分类误差和边框回归误差同Faster R-CNN，对于Mask分支，输出为K\*m\*m（每个RoI），表示K个（K类）m*m的binary mask,
    注意Mask损失只计算K个mask中 与GT的类别相同的那个mask，依赖分类分支的预测来选择相应mask（不同于FCN，FCN各个mask之间存在竞争关系）

## YOLO v3
* 只对最佳的anchor box进行边框预测，非最佳且IoU>0.5的不预测边框，同时，每个GT box只匹配一个最佳anchor box，
    这个anchor box的分类得分为1（分类使用logistic reg） 未被匹配上的anchor box仅参与分类损失计算
* 多尺度预测，根据对数据集进行的聚类，得出了9种尺寸的先验框，(10x13)，(16x30)，(33x23)，(30x61)，(62x45)，(59x119)，(116x90)，(156x198)，(373x326)，
    然后根据先验框的尺寸，将他们分配给不同尺度的特征图。v3共有3个不同尺度的feature map，深度都是255，宽高分别是13，26，52，
    其中255=3*(5+80)，3为被分配到的先验框个数，80为各类别的概率，5则是每个box的(x，y，w，h，confidence)，对小目标有一定提升

## RefineNet
* 结合one-stage和two-stage检测方法的优点，提出了ARM、ODM、TCB
* ARM：识别和删除negative anchors，减少分类器搜索空间，利用多层特征来回归bbox和前景背景二分类，优化初始值
* ODM：使用ARM的输出产生的refined anchors作为输入，融合不同层的特征，做多类别分类和bbox回归
* TCB：将ARM的输出feature map转换成ODM的输入
* 损失函数包括ARM部分的binary cls损失+回归损失，ODM部分的多分类损失和回归损失

## CornerNet
* 待总结

## M2Det
* 提出多尺度特征金字塔网络（MLFPN），包括3部分：特征融合模块FFM、细化U型模块TUM、尺度特征聚合模块SFAM
* FFM包括2部分，第一部分从backbone中取2个不同尺度的特征图作为输入，在深层特征图上进行了一次上采样操作并拼接后输出为Base Feature，
    第二部分使用Base Feature和上一个TUM输出的最大的特征图作为输入，并输出融合特征给下一个TUM
* TUM接受Base Feature/FFMv2的输出，处理并输出6个尺度的特征图金字塔，每个TUM包括5次上采样
* SFAM用于聚合8个TUM产生的多尺度输出，其沿着channel维度，将拥有相同尺度的特征图拼接，产生多尺度的特征金字塔，
    然后借鉴SENet的思想，加入channel-wise attention，更好地捕捉有用特征
* 最后在检测阶段使用2个卷积层对6个尺度的特征金字塔进行分类和边框回归

## 后续待补充
