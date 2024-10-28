import math
import scipy.io
import torch
from torch.utils.data import DataLoader

from utils.RFFIDataSet import RFFIDataSet

MODEL_PATH = "./Models"
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载MAT文件
print("Loading train set...")
label_data = scipy.io.loadmat("./DataSets/labels_wt_3.mat")
sample_data = scipy.io.loadmat("./DataSets/datas_wt_3.mat")


# 将数据重塑为样本和特征的形式
num_samples = 100
num_features = 4070
tst_samples = torch.from_numpy(sample_data["save_data"])
tst_samples = tst_samples.reshape(num_samples, 1, 2, num_features)
tst_labels = torch.from_numpy(label_data["labels"])
tst_labels = tst_labels.reshape(num_samples)

# 创建Dataset实例
dataset = RFFIDataSet(tst_samples, tst_labels)

# 创建DataLoader实例
test_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
BATCH_NUM = math.ceil(len(dataset) / BATCH_SIZE)

print("Using ", DEVICE)
print("Loading saved model...")
net = torch.load(MODEL_PATH + "/RFFI.pkl", weights_only=False).to(DEVICE)
print("Testing...")

with torch.no_grad():  # 没有求导
    correct = 0.0
    total = 0.0
    for batch_idx, (images, tst_labels) in enumerate(test_dataloader):
        net.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
        images = images.type(torch.FloatTensor)
        tst_labels = tst_labels.type(torch.LongTensor)
        outputs, fea = net(images)  ### change
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        total += tst_labels.size(0)
        correct += (predicted == tst_labels).sum().item()

    print("测试分类准确率为：%.3f%%" % (100.0 * correct / total))

