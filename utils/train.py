import torch
import torch.nn as nn
import torch.optim as optim
from LeNet.models.LeNet import LeNet
from LeNet.utils.dataloader import get_dataloaders, show_sample_images
from tqdm import tqdm


def train(trainloader, epochs, learning_rate, save_path):
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # 初始化模型
    net = LeNet().to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(epochs):
        running_loss = 0.0
        loop = tqdm(
            enumerate(trainloader),
            total=len(trainloader),
            desc=f"Epoch {epoch + 1}/{epochs}",
            colour='blue'
        )

        for i, (inputs, labels) in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (i + 1))

            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # 保存模型
    torch.save(net.state_dict(), save_path)