import math
import torch
import os
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from cnn import wt_net


DATA_PATH = "./DataSets"
MODEL_PATH = "./Models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 10
EPOCH = 40

print("Loading train set...")
# 打开 CSV 文件
data = pd.read_csv(r".\DataSets\SLF\datas.csv", header=None)


# 将数据重塑为样本和特征的形式
# 每个样本由2行组成，因此总共有500个样本
num_samples = 500
num_features = 4070
samples = data.values.reshape(num_samples, 1, 2, num_features)
labels = torch.ones(num_samples, dtype=torch.long)  # 每个样本的标签都是1


# 创建一个自定义的Dataset类
class CustomDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.tensor(self.samples[idx], dtype=torch.float32)
        label = self.labels[idx]
        return sample, label


# 创建Dataset实例
dataset = CustomDataset(samples, labels)

# 创建DataLoader实例
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
BATCH_NUM = math.ceil(len(dataset) / BATCH_SIZE)
print("Using ", DEVICE)

# 建立模型并载入设备
model = wt_net.MyCNN().to(DEVICE)
# 定义损失及优化器
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

print(
    "\n-----------------\n"
    "Num of epoch: {}\n"
    "Batch size: {}\n"
    "Num of batch: {}".format(EPOCH, BATCH_SIZE, BATCH_NUM)
)
print("-----------------\n")
print("Start training...")
# 训练
for epoch in range(EPOCH):
    print("Training epoch {}/{}".format(epoch + 1, EPOCH))
    num_correct = 0
    val_loss = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        num_correct_batch = 0
        val_loss_batch = 0
        # 注意这里的images和labels均为一个batch的图片和标签
        images = images.to(DEVICE).float()  # BATCH_SIZE*28*28
        labels = labels.to(DEVICE)  # BATCH_SIZE*1

        outputs = model(images)
        pred = torch.max(outputs, 1)[1]  # 这一步将给出每张图片的分类结果，BATCH_SIZE*1
        optimizer.zero_grad()
        loss = cost(outputs, labels)
        loss.backward()
        optimizer.step()
        val_loss_batch += loss.data
        val_loss += val_loss_batch
        num_correct_batch += (pred == labels).sum().item()
        num_correct += num_correct_batch
        print(
            "Batch {}/{}, Loss: {:.6f}, Accuracy: {:.6f}%".format(
                batch_idx + 1,
                BATCH_NUM,
                val_loss_batch / BATCH_SIZE,
                100 * num_correct_batch / BATCH_SIZE,
            )
        )
    print(
        "Epoch {}: Loss: {:.6f}, Accuracy: {:.6f}%\n".format(
            epoch + 1, val_loss / len(dataset), 100 * num_correct / len(dataset)
        )
    )
# 保存整个网络
print("Saving the model...")
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
torch.save(model, MODEL_PATH + "/MyCNN_MNIST.pkl")
