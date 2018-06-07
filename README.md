# CVDL_course_project
A CVPR PKU course project

内容：遥感图像分割

小组成员：大逸子、狗神、白哥、杨海欣

## 奋斗精神

不说太多，这个项目是10000个课程大作业（包括穆老师CVDL课程、张志华老师“深度学习”课程、李廉林老师“电磁大数据导论”课程），我们要一起**把它做好**！

## 任务说明

- 任务是遥感图像的分割。
- 目前所用模型是U-Net （[arXiv:1505.04597](https://arxiv.org/abs/1505.04597), 2015）和 Hourglass ([arXiv:1603.06937](https://arxiv.org/abs/1603.06937), 2016)。
- 数据集地址在[这里](https://files.inria.fr/aerialimagelabeling/NEW2-AerialImageDataset.zip)，训练集和测试集分别包含了五个城市的卫星照片。
- 初始阶段只区分每个像素是否属于建筑物，属于二分类问题。
- 希望后续能够挑战更复杂的数据集，终极目标是使深度学习技术服务于农业用地的监测。

## 目标计划

- 在INRIA的数据集上达到不错的结果，训练集Acc大于85\%,IoU为60\%以上。
- 如果期末前有时间，希望处理一下更复杂的数据集（参考Kaggle比赛等）。
- 找到国土规划局的土地规划方案，利用模型将其与中国目前的遥感卫星图片做对比，试图找出违反了规划要求的地方，标出地理位置和挪用土地面积。
- 写报告、做ppt…

# 已成工作

- 样本大小由400x400改为300x300。（已做）
- 做一下正负样本均衡，或者采用更简单的方法——修改loss函数。（已做）
- 已经联系上李老师那边的学姐，可以使用他们实验室的GPU资源。我下周把数据拷过去，然后多训几个epoch，看效果能不能变好…（已做）
- 模型loss的可视化做得不是特别好，cuda还存在因为memory不够而导致程序崩溃的问题，希望同志们**帮助修改**。（已做）
- 尝试多个U-Net堆叠，观察训练结果的变化。（先用两个做实验）（未做）
- 目前已经训练了几个模型，效果还有待提高（一层模型欠拟合，两层堆叠会产生过拟合），大致如图：![过拟合](https://github.com/Barak123748596/CVDL_course_project/blob/master/result.jpg)
- 设计算法可以由网络预测结果得到的概率图得到方框的算法，可以在未来grabcut中使用（但是穆老师认为grabcut所需要时间过多，可能效果不好）
- 通过CRF算法对网络模型预测结果进行修正，可以将部分误判结果修正掉，效果还挺好看的，大致如图：![效果图](https://github.com/Barak123748596/CVDL_course_project/blob/master/result.jpg)
- 目前找到了北京市等地的农田规划图，同时可以通过**水经注万能地图下载器**下载谷歌地图得到0.3米/像素的地图图片。![北京农田规划](https://github.com/Barak123748596/CVDL_course_project/blob/master/result.jpg)

## 现存问题

- 估计简单用U-Net分割效果不会特别好，之后很可能还要改进。如果有同学了解图像处理相关知识（比如预处理、传统方法分割），或者很懂深度网络调参（逸子和狗神经验不足），麻烦**喊一嗓子**，方便之后的讨论。
- 关于国土规划分布的工作还没有开展。目前要求不高，不需要上来就全国，从某个地方区县开始就可以。主要是希望证明我们“能做”。

## 阶段任务

- 需要设计算法将北京市得农田规划转换为是否应该有建筑的二维矩阵（白哥）
- 完成小组报告大纲（杨海欣）
- 做PPT等演讲必备（宋涛）
- 图片后处理（大逸子）
- 相关国内地图下载（狗神）
- 在有一定的准确率后，我们打算借鉴一下DeepLab，用空洞卷积来搭建网络，然后借助Graphcut算法进行后处理。（未做）

## 下阶段任务

- 对模型预测结果用grabcut修正。
- 想办法处理诸如“船只”、“大汽车”等经典误判。
- 找到别的数据集，想办法多分类（最多5即可）。
