import sys
import os
import yaml

# 将项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from LeNet.utils.dataloader import get_dataloaders, show_sample_images
from LeNet.utils.train import train
from LeNet.utils.test import test


def main():
    # 读取配置文件
    with open(os.path.join(root_dir, 'LeNet/configs/config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    # 提取需要的参数
    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']
    learning_rate = config['train']['learning_rate']
    data_root = config['data']['root']
    download = config['data']['download']
    save_path = config['model']['save_path']

    # 获取数据加载器
    trainloader, _, classes = get_dataloaders(batch_size, data_root, download)

    # 训练模型
    train(trainloader, epochs, learning_rate, save_path)

    # 测试模型
    test(save_path, batch_size, data_root)


if __name__ == "__main__":
    main()