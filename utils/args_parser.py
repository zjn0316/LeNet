# LeNet/utils/args_parser.py
import argparse
import yaml
import os


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LeNet训练参数配置')

    # 训练参数
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--learning_rate', type=float, help='学习率')

    # 数据参数
    parser.add_argument('--data_root', type=str, help='数据根目录')
    parser.add_argument('--download', action='store_true', help='是否下载数据集')

    # 模型参数
    parser.add_argument('--save_path', type=str, help='模型保存路径')

    # 配置文件参数
    parser.add_argument('--config', type=str, default='LeNet/configs/config.yaml', help='配置文件路径')

    return parser.parse_args()


def get_config(args, config_path=None):
    """
    获取配置，命令行参数优先级高于配置文件

    参数:
        args: 命令行参数对象
        config_path: 配置文件路径，如果为None则使用args.config

    返回:
        合并后的配置字典
    """
    # 如果未指定配置文件路径，使用参数中的路径
    if config_path is None:
        config_path = args.config

    # 读取配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 使用命令行参数覆盖配置文件中的参数
    # 训练参数
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['train']['learning_rate'] = args.learning_rate

    # 数据参数
    if args.data_root is not None:
        config['data']['root'] = args.data_root
    if args.download:
        config['data']['download'] = True

    # 模型参数
    if args.save_path is not None:
        config['model']['save_path'] = args.save_path

    return config