import torch
from tqdm import tqdm

#######定义超参数#########
epoch = 100


def test_loop(epoch):
    ##加载训练模型
    # net = torch.load('./train_model/FullConnNet__5.pth')
    # net.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for i in tqdm(range(epoch)):
            ###此处填写测试代码
            print(i)


if __name__ == '__main__':
    test_loop(epoch)
