import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F

class LightningFMNISTClassifier(pl.LightningModule):
    def __init__(self, optimizer: torch.optim.Optimizer, lr: float):
        #super(LightningFMNISTClassifier, self).__init__()
        super().__init__()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.layer_1 = torch.nn.Linear(28 * 28,392)
        self.layer_2 = torch.nn.Linear(392, 196)
        self.layer_3 = torch.nn.Linear(196, 98)
        self.layer_4 = torch.nn.Linear(98, 49)
        self.layer_5 = torch.nn.Linear(49, 10)
        self.optimizer = optimizer
        self.lr = lr
    
    def forward(self,x):
        batch_size, channels, width, height = x.size()
        
        x = x.view(batch_size, -1)
        
        x = self.layer_1(x)
        x = torch.relu(x)
        x = torch.dropout(x, 0.1, train=True)
        
        x = self.layer_2(x)
        x = torch.relu(x)
        x = torch.dropout(x, 0.1, train=True)
        
        x = self.layer_3(x)
        x = torch.relu(x)
        x = torch.dropout(x, 0.1, train=True)
        
        x = self.layer_4(x)
        x = torch.relu(x)
        
        x = self.layer_5(x)
    
        x = torch.log_softmax(x, dim=1)
        return x
    
    #тута надо lr и оптимайзер
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer, self.parameters(), lr=self.lr)
        #optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.003)
        return optimizer
        
    
    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss',loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x,y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)
        
    def predict_step(self, batch, batch_idx):
        x,y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.batch_metrics = self.accuracy.forward(logits, y)
        print(f"Metrics on batch {batch_idx}: {self.batch_metrics}")
        return loss
    
    def get_accuracy(self):
        self.accuracy_score = self.accuracy.compute()
        print(f"Metrics on set: {self.accuracy_score}")
    