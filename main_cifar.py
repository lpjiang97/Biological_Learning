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

from models.model import BPNet
from trainer.trainer import train, test
from cmdin import args

# data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # CIFAR 10 stats
])
batch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

# model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# one or two layers
two_layer = args.mode == "two"
#model = BPNet(32 * 32, 3, 2000, 10, two_layer=two_layer).to(device) # CIFAR 10 
model = BPNet(32 * 32, 3, 100, 10, two_layer=two_layer).to(device) # CIFAR 10 

# training
E = args.epochs
lr = args.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.2)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.5)

# record
i = 0
while True:
    run_base_dir = pathlib.Path("cifar_logs") / f"{args.sess}_try={str(i)}"
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

criteria = nn.CrossEntropyLoss()

# main training loop
for epoch in tqdm(range(E), desc="Epoch", total=args.epochs, dynamic_ncols=True):
    # train
    train_loss, train_accuracy = train(model, train_loader, optimizer, criteria, device, train_writer, epoch) 
    # test
    test_loss, test_accuracy = test(model, test_loader, criteria, device, test_writer, epoch) 
    # log to file
    train_csv_writer.writerow([train_loss, train_accuracy])  
    test_csv_writer.writerow([test_loss, test_accuracy])  
    # save checkpoint
    if epoch % 10 == 9:
        torch.save(model.state_dict(), run_base_dir / f"epoch_{epoch+1}.pt") 
    scheduler1.step()
    scheduler2.step()
torch.save(model.state_dict(), run_base_dir / f"epoch_{epoch+1}.pt") 
f_train.close()
f_test.close()
