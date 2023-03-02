import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision import datasets, transforms

class FMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_dir = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])
        self.batch_size = batch_size
        
    def prepare_data(self):
        # download
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)
        
    def setup(self, stage):
        if stage == "fit":
            fmnist_full = FashionMNIST(self.data_dir, train=True, transform=self.transform)
            print(len(fmnist_full))
            self.fmnist_train, self.fmnist_val = random_split(fmnist_full, [55000, 5000])
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.fmnist_test = FashionMNIST(self.data_dir, train=False, transform=self.transform)
        if stage == "predict":
            self.fmnist_predict = FashionMNIST(self.data_dir, train=False, transform=self.transform)
    
    # батч надо
    def train_dataloader(self):
        return DataLoader(self.fmnist_train, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.fmnist_val, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.fmnist_test, self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.fmnist_predict, self.batch_size)