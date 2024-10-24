import math
import torch
import torch.nn as nn
import os
import scipy.io
from tqdm import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from cnn import rsnet
from utils.RFFIDataSet import RFFIDataSet

MODEL_PATH = "./Models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCH = 100

# 加载MAT文件
print("Loading train set...")
label_data = scipy.io.loadmat("./DataSets/labels_wt_2.mat")
sample_data = scipy.io.loadmat("./DataSets/datas_wt_2.mat")


# 将数据重塑为样本和特征的形式
num_samples = 200
num_features = 4070
samples = torch.from_numpy(sample_data["save_data"])
samples = samples.reshape(num_samples, 1, 2, num_features)
labels = torch.from_numpy(label_data["labels"])
labels = labels.reshape(num_samples)


# 创建Dataset实例
dataset = RFFIDataSet(samples, labels)

# 创建DataLoader实例
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
BATCH_NUM = math.ceil(len(dataset) / BATCH_SIZE)
print("Using ", DEVICE)

# 建立模型并载入设备
net = rsnet.rsnet34().to(DEVICE)
# 定义损失及优化器
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

print(
    "\n-----------------\n"
    "Num of epoch: {}\n"
    "Batch size: {}\n"
    "Num of batch: {}".format(EPOCH, BATCH_SIZE, BATCH_NUM)
)
print("-----------------\n")
print("Start training...")

pbar = tqdm(range(EPOCH))

Accs = []
Losss = []
loss_function = nn.CrossEntropyLoss()  ## 交叉熵损失函数

# 训练
for epoch in pbar:
    # print("Training epoch {}/{}".format(epoch + 1, EPOCH))
    pbar.set_description("Training epoch")
    net.train()
    val_loss = 0.0  # 损失数量
    num_correct = 0.0  # 准确数量
    total = 0.0  # 总共数量
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        length = len(train_dataloader)
        images = images.type(torch.FloatTensor)
        labels = labels.type(torch.LongTensor)

        optimizer.zero_grad()
        outputs, fea = net(images)  ### change
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        num_correct += (predicted == labels).sum().item()

    pbar.set_postfix(
        Accuracy="{:.6f}%".format(100 * num_correct / len(dataset)),
        Loss="{:.6f}".format(val_loss / len(dataset)),
    )
    Accs.append(num_correct / len(dataset))
    Losss.append(val_loss / len(dataset))
# 保存整个网络
print("Saving the model...")
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
torch.save(net, MODEL_PATH + "/RFFI.pkl")

# 绘制Accs的折线图
fig, ax1 = plt.subplots()
ax1.plot(range(len(Accs)), Accs, marker="o", label="Accuracy", color="blue")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# 创建共享X轴的第二个Y轴并绘制Losss的折线图
ax2 = ax1.twinx()
ax2.plot(range(len(Losss)), Losss, marker="x", label="Loss", color="red")
ax2.set_ylabel("Loss", color="red")
ax2.tick_params(axis="y", labelcolor="red")

# 添加标题和图例
plt.title("Accuracy and Loss of Each Epoch")
fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

# 显示图表
plt.grid(True)
plt.show()
