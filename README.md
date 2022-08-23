# Robotic-arm

## 机械臂小组项目

- 环境：python3.6+pytorch，或许还需要一些辅助工具包如numpy，pandas，tqdm等等
#### 各文件作用:

- FullConnNet.py:定义全连接神经网络模型的代码
- train.py:训练该模型的代码
- test.py:测试该模型的代码
- train_model:存放训练好的模型
- demo.py:完整的神经网络demo，拟合sinx函数，使用的是gpu训练，若是cpu需要修改部分代码

#### 需要完成的工作:

将数据导入模型训练并且验证结果，数据的操作可单独写在dataset.py(未创建)文件中
