import os
import csv
import json
import pathlib
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models.model import SparseNet
from models.lossses import OnehotLoss
from trainer.trainer import train, test
from cmdin import args

# data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST stats
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # CIFAR 10 stats
])
batch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

# model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# one or two layers
two_layer = args.mode == "two"
#model = BPNet(28 * 28, 1, 2000, 10, two_layer=two_layer).to(device) # MNIST
model = SparseNet(2000, 28 * 28, r_lr=50, lmda=0.001).to(device)

# training
E = args.epochs
lr = args.lr
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# record
i = 0
while True:
    run_base_dir = pathlib.Path("sc_logs") / f"{args.sess}_try={str(i)}"
    if not run_base_dir.exists():
        os.makedirs(run_base_dir)
        break
    i += 1
with open(run_base_dir / "args.json", 'w') as f:
    json.dump(vars(args), f)
train_writer = SummaryWriter(run_base_dir / "train")
test_writer = SummaryWriter(run_base_dir / "test")
f_train = open(run_base_dir / "loss_train.csv", "w")
f_test = open(run_base_dir / "loss_teset.csv", "w")
train_csv_writer = csv.writer(f_train)
test_csv_writer = csv.writer(f_test)

criteria = nn.MSELoss()

# main training loop
for epoch in tqdm(range(E), desc="Epoch", total=args.epochs, dynamic_ncols=True):
    # train
    train_loss, train_accuracy = train(model, train_loader, optimizer, criteria, device, train_writer, epoch, sc=True) 
    # test
    test_loss, test_accuracy = test(model, test_loader, criteria, device, test_writer, epoch, sc=True) 
    # log to file
    train_csv_writer.writerow([train_loss, train_accuracy])  
    test_csv_writer.writerow([test_loss, test_accuracy])  
    # save checkpoint
    if epoch % 10 == 9:
        torch.save(model.state_dict(), run_base_dir / f"epoch_{epoch+1}.pt") 
torch.save(model.state_dict(), run_base_dir / f"epoch_{epoch+1}.pt") 
f_train.close()
f_test.close()
