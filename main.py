import sys
import os
import yaml

# 将项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from LeNet.utils.dataloader import get_dataloaders, show_sample_images
from LeNet.utils.args_parser import parse_args,get_config
from LeNet.utils.train import train
from LeNet.utils.test import test



def main():
    # 解析命令行参数
    args = parse_args()

    # 获取配置，命令行参数优先级高于配置文件
    config = get_config(args, os.path.join(root_dir, args.config))
    batch_size = config['train']['batch_size']
    data_root = config['data']['root']
    download = config['data']['download']
    epochs = config['train']['epochs']
    learning_rate = config['train']['learning_rate']
    save_path = config['model']['save_path']

    # 获取数据加载器
    trainloader, _, classes = get_dataloaders(batch_size, data_root, download)

    # 训练模型
    train(trainloader, epochs, learning_rate, save_path)

    # 测试模型
    test(save_path, batch_size, data_root)


if __name__ == "__main__":
    main()