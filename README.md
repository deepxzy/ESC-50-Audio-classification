# ESC-10数据集的音频分类

本文选取了ESC-10数据集作为音频分类的数据集，选取了四种方法进行分类，包括支持向量机，决策树，多层感知机，循环神经网络。最终支持向量机取得了最优的结果（0.825）。


### 多层感知机（MLP）

学习率设置为0.001，优化算法选择Adam，损失函数为交叉熵函数，训练10个epoch的损失函数变化如下所示：

训练集上的损失函数:

![](C:\Users\xzy123\Desktop\work\train_loss1.png)

测试集上的损失函数:

![](C:\Users\xzy123\Desktop\work\test_loss1.png)

测试集上的准确率:

![]([ESC-50-Audio-classification/checkpoints/mlp/test_accury.png](https://github.com/deepxzy/ESC-50-Audio-classification/blob/master/checkpoints/mlp/test_accury.png))



### 循环神经网络（RNN）

训练集上的损失函数:

![](C:\Users\xzy123\Desktop\work\train_loss.png)

测试集上的损失函数:

![](C:\Users\xzy123\Desktop\work\test_loss.png)

测试集上的准确率:

![](C:\Users\xzy123\Desktop\work\test_accury.png)

## 总结

|          |                   支持向量机                    |                     决策树                      |                   多层感知机                    | 循环神经网络 |
| :------: | :---------------------------------------------: | :---------------------------------------------: | :---------------------------------------------: | :----------: |
|   特征   | mfcc，chroma，melspectrogram，contrast，tonnetz | mfcc，chroma，melspectrogram，contrast，tonnetz | mfcc，chroma，melspectrogram，contrast，tonnetz |     mfcc     |
| 特征维度 |                    （1，45）                    |                    （1，5）                     |                    （1，45）                    | （431, 20）  |
|  准确率  |                      0.825                      |                      0.650                      |                      0.762                      |    0.750     |

## 附录

下面展示的实验程序结果原图：

支持向量机：

![](C:\Users\xzy123\Desktop\work\附录\svm.png)

决策树：

![](C:\Users\xzy123\Desktop\work\附录\clf.png)

多层感知机：

![](C:\Users\xzy123\Desktop\work\附录\mlp.png)

循环神经网络：

![](C:\Users\xzy123\Desktop\work\附录\rnn.png)
