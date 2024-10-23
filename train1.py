import math
import torch
import torch.nn as nn
import os
import scipy.io
from tqdm import *
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from cnn import rsnet


DATA_PATH = "./DataSets"
MODEL_PATH = "./Models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCH = 10

# 加载MAT文件
print("Loading train set...")
label_data = scipy.io.loadmat("./DataSets/SLF/labels.mat")
sample_data = scipy.io.loadmat("./DataSets/SLF/datas.mat")


# 将数据重塑为样本和特征的形式
# 每个样本由2行组成，因此总共有500个样本
num_samples = 600
num_features = 4070
samples = torch.from_numpy(sample_data["save_data"])
samples = samples.reshape(num_samples, 1, 2, num_features)
labels = torch.from_numpy(label_data["labels"])
labels = labels.reshape(num_samples)


# 创建一个自定义的Dataset类
class CustomDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # sample = torch.tensor(self.samples[idx], dtype=torch.float32)
        sample = self.samples[idx]
        label = self.labels[idx]
        return sample, label


# 创建Dataset实例
dataset = CustomDataset(samples, labels)

# 创建DataLoader实例
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
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
    for i,(X,y) in enumerate(dataloader):
        length = len(dataloader)
        X = X.type(torch.FloatTensor)
        y = y.type(torch.LongTensor)

        optimizer.zero_grad()
        outputs,fea = net(X)  ### change
        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()

        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        num_correct += (predicted == y).sum().item()
        # correct += predicted.eq(y.data).cpu().sum()
        # print('[epoch:%d, iter:%d/%d] Loss: %.03f | Acc: %.3f%% '
        #       % (epoch + 1, (i + 1), length, val_loss / (i + 1), 100. * num_correct / total))

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
torch.save(net, MODEL_PATH + "/MyCNN_MNIST.pkl")

# 绘制Accs的折线图  
fig, ax1 = plt.subplots()  
ax1.plot(range(len(Accs)), Accs, marker="o", label="Accuracy", color="blue")  
ax1.set_xlabel("Epoch")  
ax1.set_ylabel("Accuracy", color="blue")  
ax1.tick_params(axis='y', labelcolor="blue")  
  
# 创建共享X轴的第二个Y轴并绘制Losss的折线图  
ax2 = ax1.twinx()  
ax2.plot(range(len(Losss)), Losss, marker="x", label="Loss", color="red")  
ax2.set_ylabel("Loss", color="red")  
ax2.tick_params(axis='y', labelcolor="red")  
  
# 添加标题和图例  
plt.title("Accuracy and Loss of Each Epoch")  
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)  
  
# 显示图表  
plt.grid(True)  
plt.show()