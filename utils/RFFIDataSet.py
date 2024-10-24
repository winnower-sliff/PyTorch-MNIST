from torch.utils.data import Dataset


class RFFIDataSet(Dataset):
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
