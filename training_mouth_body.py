import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.datasets import DatasetFolder
from torch import distributed as dist
import torchvision
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import time
import skvideo.io, skvideo.utils
from skimage.color import rgb2lab
from sklearn import metrics
from vidaug import augmentors as va
import random
from pytorchvideo.transforms import RandAugment
import pickle

#Parameter
lr = 0.00001
batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
error = nn.CrossEntropyLoss()

#To change
pthSave = "./last4_body.pth"
tensorboardSave = "last4/bodyh"
trainPath = "./ambiguous_body_train"
valPath = "./ambiguous_body_test"

max_frames =28

class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(0.5),
            nn.BatchNorm3d(8),

            nn.Conv3d(8, 16, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(0.5),
            nn.BatchNorm3d(16),

            nn.Conv3d(16, 32, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(0.5),
            nn.BatchNorm3d(32),

        )
        self.fc = nn.Sequential(
            nn.Linear(2*512*max_frames, 500),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(500, 12),
        )
        #1728
        self.lstm = nn.GRU(1728, 1000, 1, bidirectional=True)
        self.lstm2 = nn.GRU(2000, 512, 1, bidirectional=True)

        self.batch = nn.BatchNorm3d(3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.batch(x)
        # B C T H W
        x = self.conv(x)
        # T B C H W
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # T B C*H*W
        x = x.view(x.size(0), x.size(1), -1)
        #x = F.log_softmax(x, dim=-1)
        self.lstm.flatten_parameters()
        self.lstm2.flatten_parameters()
        # T B Features
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = x.permute(1,0,2).contiguous()
	# B T Features
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def init_process_group():
    """
    Join the process group and return whether this is the rank 0 process,
    the CUDA device to use, and the total number of GPUs used for training.
    """
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    dist.init_process_group('nccl')
    return rank == 0, torch.device(f'cuda:{local_rank}'), num_gpus

randaug = RandAugment(4,3)

def loaderAug(path):
    # Read Video
    data = torchvision.io.read_video(path)[0]


    # Dimension: Time, Channels, Height, Width
    data = data.permute(0, 3, 1, 2)

    data = randaug(data)

    # Normalisation
    x = data / 255.0
    # Concatenate last frame until 28 frames is reached
    if x.size(0) < max_frames:
        conframe = x[-1].unsqueeze(0)
        while conframe.size(0) + x.size(0) != max_frames:
            conframe = torch.cat([conframe, x[-1].unsqueeze(0)], 0)
        x = torch.cat([x, conframe], 0)
    if x.size(0) != max_frames:
        print(x.size(0))
        print(path)
        #os.remove(path)
        pass
    else:
        color = []
        for frame in x:
            frame = frame.permute(1,2,0)
            color.append(torch.from_numpy(rgb2lab(frame.numpy())).permute(2,0,1))
        color = torch.stack(color)
        # C T H W
        color = color.permute(1, 0, 2, 3)
        return color


def loader(path):
    data = torchvision.io.read_video(path)[0]

    # Dimension: Time, Channels, Height, Width
    data = data.permute(0, 3, 1, 2)

    x = data / 255.0
    # Concat last frame until 28 frames is reached
    if x.size(0) < max_frames:
        conframe = x[-1].unsqueeze(0)
        while conframe.size(0) + x.size(0) != max_frames:
            conframe = torch.cat([conframe, x[-1].unsqueeze(0)], 0)
        x = torch.cat([x, conframe], 0)
    if x.size(0) != max_frames:
        print(x.size(0))
        print(path)
        #os.remove(path)
        pass
    else:
        color = []
        for frame in x:
            frame = frame.permute(1,2,0)
            color.append(torch.from_numpy(rgb2lab(frame.numpy())).permute(2,0,1))
        color = torch.stack(color)
        # C T H W
        color = color.permute(1, 0, 2, 3)
        return color


train = DatasetFolder(trainPath, loader=loaderAug, extensions=("avi"))
val = DatasetFolder(valPath, loader=loader, extensions=("avi"))
klasse = val.classes
klasssen = dict()
print(klasse)
print()
print(klassen)

for b in val.class_to_idx:
    klassen[val.class_to_idx[b]] = b

if __name__ == "__main__":
    print(23)
    is_rank0, device, num_gpus = init_process_group()
    torch.cuda.set_device(device)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train, num_replicas=num_gpus, rank=int(os.getenv('RANK', 0)))
    val_sampler = torch.utils.data.distributed.DistributedSampler(val, num_replicas=num_gpus, rank=int(os.getenv('RANK', 0)))
    
    trainloader = DataLoader(train, shuffle=False, batch_size=64, num_workers=4, sampler=train_sampler)
    valloader = DataLoader(val, shuffle=False, batch_size=1, num_workers=4, sampler=val_sampler)
    
    Model = Netz()
    Model.to(device)
    Model = nn.parallel.DistributedDataParallel(Model)
    
    #Model.load_state_dict(torch.load(pthSave))

    optimizer = torch.optim.Adam(Model.parameters(), lr=lr)
    trainAcc = []
    valAcc = []

    Model.train()

    def test(dataloader):
        with torch.no_grad():
            Model.eval()
            Model.to(device)
            correct = 0
            total = 0

            actual = []
            pred = []
            classes= []

            for images, labels in dataloader:
                images, labels = Variable(images).to(device, dtype=torch.float), Variable(labels).to(device)
                output = Model.forward(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
               
                #for topk accuracy batchsize must be 1!
                #_, topk_indices = torch.topk(output, 2)
                #print(labels, topk_indices)
                #correct += int(labels[0] in topk_indices)
                
                #print(output.shape, labels.shape, predicted.shape)

                #actual.append(klassen[predicted.item()])
                #if labels.item() not in classes:
                    #classes.append(klassen[labels.item()])
                #pred.append(klassen[labels.item()])

        #with open('body_preds.pickle', 'wb') as handle:
        #    pickle.dump([pred, actual], handle)

        #cf = confusion_matrix(pred, actual,normalize="true")
        
        #print(metrics.classification_report(actual, pred, digits=4))
        #fig, ax = plt.subplots(figsize=(10,10))
        #sns.heatmap(cf, annot=True, fmt='.2f', xticklabels=klasse, yticklabels=klasse)
        #plt.ylabel('Actual')
        #plt.xlabel('Predicted')        
        #plt.savefig("confusionbody_04.png")
        Model.train()
        return 100 * (correct / total)

    # ---Training
    def train(model, trainloader):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        error = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.883276740237691, 0.9061969439728353, 0.8972835314091681, 0.967741935483871, 0.9711375212224108, 0.8972835314091681,
                                            0.8743633276740238, 0.883276740237691, 0.9711375212224108, 0.9061969439728353, 0.8743633276740238, 0.967741935483871]).to(device))
        model.train()  
        besteVal = 0  

        overfitted = 0
        writer = SummaryWriter(tensorboardSave)

        for i in range(0, 5000):
            correct = 0
            total = 0
            a = time.time()
            for batch_idx, (images, labels) in enumerate(trainloader):
                images = Variable(images).to(device, dtype=torch.float)
                labels = Variable(labels).to(device)

                optimizer.zero_grad()
                # prediction
                output = model.forward(images)
                # Loss & Backprogagation
                loss = error(output, labels)
                loss.backward()
                optimizer.step()
                # Accuracy
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            aktuelleVal = test(valloader)
            # save weights with of epoch with highest accuracy
            if aktuelleVal > besteVal:
                besteVal = aktuelleVal
                torch.save(Model.state_dict(), pthSave)
                #print(besteVal)
            print("Epoche", i, str(total), "/", len(trainloader) * batch_size, "Loss: ", loss.data.item(), " ",
                  "Genauigkeit: ", str(correct / total*100), "Val-Genauigkeit: ", str(aktuelleVal))
            writer.add_scalars("Genauigkeit", {"Train": np.float(str(correct / total*100)), "Val": np.float(aktuelleVal)}, i)
            writer.flush()

            trainAcc.append((correct / total) * 100)
            valAcc.append(aktuelleVal)

        print("Bestes Ergebnis:", besteVal)
        writer.close()

    # Training
    train(Model, trainloader)

    #print(test(valloader))

