import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
import torchvision.models as models


class MyNet(pl.LightningModule):

    def __init__(self, hparams, train_set=None, val_set=None, test_set=None):
        super().__init__()

        # set hyperparams
        self.save_hyperparameters(hparams)

        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}

        # assert self.hparams["activation_func"] in ['Tanh', 'LeakyReLU']
        # if self.hparams["activation_func"] == 'Tanh':
        #     self.activation_func = nn.Tanh()
        # elif self.hparams["activation_func"] == 'LeakyReLU':
        #     self.activation_func = nn.LeakyReLU()

        # self.model = models.resnet18(pretrained=True)
        # num_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_features, 4)

        self.model = models.mobilenet_v2(pretrained=True)
        # self.classifier = nn.Linear(1280, 4)
        self.model.classifier = nn.Linear(1280, 4)

    def forward(self, x):

        # feed x into model
        x = self.model(x)
        # x = self.classifier(x)

        return x

    def general_step(self, batch, batch_idx, mode):
        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        loss = F.cross_entropy(out, targets)

        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        n_total = len(targets)
        return loss, n_correct, n_total

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        length = sum([x[mode + '_n_total'] for x in outputs])
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / length
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "train")
        self.log('loss', loss)
        return {'loss': loss, 'train_n_correct': n_correct, 'train_n_total': n_total}

    def validation_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "val")
        self.log('val_loss', loss)
        return {'val_loss': loss, 'val_n_correct': n_correct, 'val_n_total': n_total}

    def test_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct, 'test_n_total': n_total}

    def validation_epoch_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        self.log('val_loss', avg_loss)
        self.log('val_acc', acc)
        return {'val_loss': avg_loss, 'val_acc': acc}

    def configure_optimizers(self):
        optim = None

        if self.hparams["optimizer"] == 'Adam':
            optim = torch.optim.Adam(self.model.parameters(), lr=self.hparams["learning_rate"])
        elif self.hparams["optimizer"] == 'SGD':
            optim = torch.optim.SGD(self.model.parameters(), lr=self.hparams["learning_rate"], momentum=0.9)

        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20000, gamma=0.9)

        # return {"optimizer": optim, "lr_scheduler": lr_scheduler}
        return optim

    def getTestAcc(self, loader):

        self.model.eval()
        self.model = self.model.to(self.device)

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()

        self.log('test_acc', acc)

        return preds, acc

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], batch_size=self.hparams['batch_size'])

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], batch_size=self.hparams['batch_size'])

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location='cuda:0'), strict=False)