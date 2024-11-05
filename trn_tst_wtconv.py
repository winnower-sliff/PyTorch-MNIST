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
BATCH_SIZE = 256
EPOCH = 100

# 加载MAT文件
print("Loading train & test set...")
# trnFN=5
# tstFN=6
# trn_label_data = scipy.io.loadmat("./DataSets/labels_wt_"+str(trnFN)+".mat")
# trn_sample_data = scipy.io.loadmat("./DataSets/datas_wt_"+str(trnFN)+".mat")
# tst_label_data = scipy.io.loadmat("./DataSets/labels_wt_"+str(tstFN)+".mat")
# tst_sample_data = scipy.io.loadmat("./DataSets/datas_wt_"+str(tstFN)+".mat")

# # 将数据重塑为样本和特征的形式
# wt_lv_length = 4070
# trn_samples = torch.from_numpy(trn_sample_data["save_data"])
# trn_samples = trn_samples.reshape(-1, 1, 2, wt_lv_length)
# trn_labels = torch.from_numpy(trn_label_data["labels"])
# trn_labels = trn_labels.reshape(-1)
# tst_samples = torch.from_numpy(tst_sample_data["save_data"])
# tst_samples = tst_samples.reshape(-1, 1, 2, wt_lv_length)
# tst_labels = torch.from_numpy(tst_label_data["labels"])
# tst_labels = tst_labels.reshape(-1)

# # 创建Dataset实例
# trn_dataset = RFFIDataSet(trn_samples, trn_labels)
# tst_dataset = RFFIDataSet(tst_samples, tst_labels)

all_label_data = scipy.io.loadmat("./DataSets/labels_wt_8.mat")
all_sample_data = scipy.io.loadmat("./DataSets/datas_wt_8.mat")

wt_lv_length = 4070
wt_lv_width = 2
channels = 1
raw_samples = torch.from_numpy(all_sample_data["save_data"])
raw_samples = raw_samples.reshape(-1, channels, wt_lv_width, wt_lv_length)
raw_labels = torch.from_numpy(all_label_data["labels"])
raw_labels = raw_labels.reshape(-1)

# 强调变换高点
change_samples=torch.clone(raw_samples)
change_samples[change_samples<0.01]=0
combined_samples = torch.cat((raw_samples, change_samples), dim=2)  # 沿着通道维度（第二个维度）拼接  
combined_samples.shape
change_labels = raw_labels - 1

all_dataset = RFFIDataSet(combined_samples, change_labels)

## 80%用于训练
train_size = int(len(all_dataset) * 0.8)
test_size = len(all_dataset) - train_size
trn_dataset, tst_dataset = torch.utils.data.random_split(
    all_dataset, [train_size, test_size]
)

# 创建DataLoader实例
trn_dataloader = DataLoader(trn_dataset, batch_size=BATCH_SIZE, shuffle=True)
tst_dataloader = DataLoader(tst_dataset, batch_size=BATCH_SIZE, shuffle=True)

TRN_BATCH_NUM = math.ceil(len(trn_dataset) / BATCH_SIZE)
TST_BATCH_NUM = math.ceil(len(tst_dataset) / BATCH_SIZE)
print("Using ", DEVICE)

change_samples = 1
if change_samples == 1:
    print("Creating new model...")
    net = rsnet.rsnet34(num_classes=10, in_channels=channels).to(DEVICE)
else:
    print("Loading saved model...")
    net = torch.load(MODEL_PATH + "/RFFI.pkl", weights_only=False).to(DEVICE)

# 定义损失及优化器
loss_function = nn.CrossEntropyLoss()  ## 交叉熵损失函数
optimizer = torch.optim.Adam(net.parameters())

print(
    "\n-----------------\n"
    "Num of epoch: {}\n"
    "Batch size: {}\n"
    "Num of train batch: {}\n"
    "Num of test batch: {}\n"
    "-----------------\n".format(EPOCH, BATCH_SIZE, TRN_BATCH_NUM, TST_BATCH_NUM)
)

trn_acc_perepoch = []
loss_perepoch = []
tst_acc_perepoch = []

print("Start training & testing...\n")
pbar = tqdm(range(EPOCH))
for epoch in pbar:
# for epoch in range(EPOCH):
    pbar.set_description("Epoch")
    print()

    net.train()
    val_loss = 0.0  # 损失数量
    num_correct = 0.0  # 准确数量
    total = 0.0  # 总共数量
    for _, (images, labels) in enumerate(trn_dataloader):
        length = len(trn_dataloader)
        images = images.type(torch.FloatTensor).to(DEVICE)
        labels = labels.type(torch.LongTensor).to(DEVICE)

        optimizer.zero_grad()
        outputs = net(images)  ### change
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        num_correct += (predicted == labels).sum().item()
    trn_accuracy = 100 * num_correct / len(trn_dataset)
    avg_loss = 100 * val_loss / len(trn_dataset)

    with torch.no_grad():  # 禁用梯度计算，以加速推理并减少内存消耗
        correct = 0.0
        total = 0.0
        for _, (images, tst_labels) in enumerate(tst_dataloader):
            net.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
            images = images.type(torch.FloatTensor).to(DEVICE)
            tst_labels = tst_labels.type(torch.LongTensor).to(DEVICE)
            outputs = net(images)  ### change
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            total += tst_labels.size(0)
            correct += (predicted == tst_labels).sum().item()
        tst_accuracy = 100.0 * correct / total

    # 统计数据并展示
    pbar.set_postfix(
        TrnAcc="{:.3f}%".format(trn_accuracy),
        Loss="{:.3f}%".format(avg_loss),
        TstAcc="{:.3f}%".format(tst_accuracy),
    )
    # print(f"Epoch = {epoch}",
    #     "\tLoss = {:.3f}%".format(avg_loss),
    #     "\tTrnAcc = {:.3f}%".format(trn_accuracy),
    #     "TstAcc = {:.3f}%".format(tst_accuracy)
    # )
    trn_acc_perepoch.append(trn_accuracy)
    loss_perepoch.append(avg_loss)
    tst_acc_perepoch.append(tst_accuracy)
    if tst_accuracy>90:
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        torch.save(net, MODEL_PATH + f"/RFFI_{epoch}.pkl")
# 保存整个网络
# print("\nSaving the model...")
# if not os.path.exists(MODEL_PATH):
#     os.makedirs(MODEL_PATH)
# torch.save(net, MODEL_PATH + "/RFFI.pkl")

# 绘制trn_acc_perepoch的折线图
print("Plotting results... ")
fig, ax1 = plt.subplots()
ax1.plot(
    range(len(trn_acc_perepoch)),
    trn_acc_perepoch,
    # marker="o",
    label="Train Set Accuracy",
    color="blue",
)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# 绘制loss_perepoch的折线图
ax2 = ax1.twinx()
ax2.plot(
    range(len(loss_perepoch)),
    loss_perepoch,
    # marker="x",
    label="Loss",
    color="red",
)
ax2.set_ylabel("Loss", color="red")
ax2.tick_params(axis="y", labelcolor="red")

# 绘制tst_acc_perepoch的折线图
ax1.plot(
    range(len(tst_acc_perepoch)),
    tst_acc_perepoch,
    # marker="s",
    label="Test Set Accuracy",
    color="green",
)

# 添加标题和图例
plt.title("Accuracy and Loss of Each Epoch")
fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

# 显示图表
plt.grid(True)
plt.show()
