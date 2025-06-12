import torch
from LeNet.models.LeNet import LeNet



def test(save_path, batch_size, data_root):
    # 加载数据
    from LeNet.utils.dataloader import get_dataloaders
    _, testloader, classes = get_dataloaders(batch_size, data_root, download=False)

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # 加载模型
    net = LeNet().to(device)
    try:
        net.load_state_dict(torch.load(save_path, map_location=device))
        print(f"成功加载模型权重: {save_path}")
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        return

    # 评估模型
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

    # 打印每个类别的准确率
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(min(batch_size, 4)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f'Accuracy of {classes[i]:5s} : {100 * class_correct[i] / class_total[i]:2f}%')