# Temporal Action Detection
## SST: Single-Stream Temporal Action Proposals

### 1.网络结构

![1543482634204](/home/liweijie/.config/Typora/typora-user-images/1543482634204.png)

以16帧为间隔将原始视频划分为小段，称为单位时间步。对每一时间步采用3D卷积网络提取较为底层的特征。这些特征被作为循环神经网络的输入。循环神经网络可以累积短时序的特征，从而根据更长时间跨度上获得的信息给出proposals。循环神经网络在每一个时间步的输出为以当前时间步为结尾的k个不同长度proposal的置信度。

### 2.训练

为了使网络可以根据长时间的上下文信息生成proposal，训练样本的长度远远大于proposal的最大长度。作者采用固定长度和较小的步长在原始视频上进行密集地采样。训练样本的标签形式被设计为与网络输出相似，对于每一个时间步，其标签为一个k维向量，每一元素在对应长度的proposal与groud truth的tIOU值大于0.5时为1，否则为0。损失函数采用交叉熵，其计算公式如下。
$$
L(c,t,X,y) = -\sum_{j=1}^{k} w_0^j y_t^j \log c_t^j+w_1^j(1-y_t^j)\log (1-c_t^j)
$$

$$
L_{train}=\sum_{(X,y)\in\chi}\sum_{t=1}^{T_w}L(c,t,X,y)
$$



### 3.问题

![1543490704693](/home/liweijie/.config/Typora/typora-user-images/1543490704693.png)

可能得到“umbrella proposal”，即比若干较短的ground truth时间跨度都要大的proposal

## CDC : Convolutional-De-Convolutional Networks for Precise Temporal Action Localization in Untrimmed Videos

### 1.网络结构

解决动作分类的常见深度学习方法如 C3D, Two-Stream Network 等给出的是视频级别的分类结果，而在动作检测任务中我们需要给出更细粒度的分类。文章中，作者受到图像语义分割问题中使用的卷积反卷积网络的启发，设计了CDC网络得到frame-level的分类结果。作者提出了CDC卷积核，使得在卷积过程中在空间上进行降采样的同时在时间上完成上采样。经过多个CDC层后，输出张量的时间维度恢复到与输入相同，也就得到了每一帧所对应的动作标签。其网络结构如下图所示

![1543496259035](/home/liweijie/.config/Typora/typora-user-images/1543496259035.png)

### 2.问题

得出的结果来自较短时间内所积累的信息，对于时间跨度较大的动作，可能无法根据充足的上下文信息作出判断







