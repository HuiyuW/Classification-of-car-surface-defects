from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
from dataset import CustomImageDataset
import torchvision.transforms as transforms
from models import MyNet
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

# introduce transformations
my_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# init dataset
train_dataset = CustomImageDataset('train_dataset.csv', 'Annotated_images', transform=my_transform)
val_dataset = CustomImageDataset('val_dataset.csv', 'Annotated_images', transform=my_transform)
test_dataset = CustomImageDataset('test_dataset.csv', 'Annotated_images', transform=my_transform)

# NN hyper-parameters for training
net_hparams = {
    # "output_size": 2,
    # "activation_func": 'LeakyReLU',  # choose in ['Tanh', 'LeakyReLU']
    "optimizer": "Adam",
    "learning_rate": 5e-4,
    "batch_size": 64
}


# load NN model
my_model = MyNet(net_hparams, train_set=train_dataset, val_set=val_dataset, test_set=test_dataset)

# set logger name and path
my_logger = TensorBoardLogger(save_dir='lightning_logs', name="test")

# Init ModelCheckpoint callback, monitoring 'val_loss'
checkpoint_callback = ModelCheckpoint(monitor="val_loss")

# early_stop_callback = EarlyStopping(
#    monitor='val_loss',
#    patience=10,
#    verbose=False,
#    mode='min'
# )

# setup trainer
trainer = pl.Trainer(
    max_epochs=20,
    logger=my_logger,
    # log_every_n_steps=2,
    # check_val_every_n_epoch=25,
    gpus=1 if torch.cuda.is_available() else None,
    callbacks=[checkpoint_callback],
    # resume_from_checkpoint="some/path/to/my_checkpoint.ckpt" # recover from a checkpoint
)


if __name__ == '__main__':

    # my_model.load_from_checkpoint('lightning_logs/test/version_2/checkpoints/epoch=2-step=36.ckpt')

    # start training
    # trainer.fit(my_model)

    # test on the trained dataset
    trainer.test(my_model)

    # load pretrained NN from record and init a test loader for evaluating
    model_test = MyNet(net_hparams, train_set=train_dataset, val_set=val_dataset, test_set=test_dataset)
    model_new = model_test.load_from_checkpoint('lightning_logs/test/version_5/checkpoints/epoch=1-step=24.ckpt')
    #
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    # if you want to plot the comparison between truths and predictions, uncomment the following lines
    pred, acc = model_new.getTestAcc(loader=test_loader)
    print(acc)

    # if you want to record the acc, uncomment the following lines
    writer1 = SummaryWriter(log_dir='lightning_logs')
    writer1.add_scalar(tag='test_acc', scalar_value=acc)
    writer1.close()