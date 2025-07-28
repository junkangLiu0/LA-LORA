import os
import torch

from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import SubsetRandomSampler, random_split, DataLoader
import time
import random
from copy import deepcopy
import ray
import argparse
from tensorboardX import SummaryWriter
from dirichlet_data import data_from_dirichlet
from sam import SAM
from model import swin_tiny_patch4_window7_224 as swin_tiny
from model import swin_small_patch4_window7_224 as swin_small
from model import swin_large_patch4_window7_224_in22k as swin_large
from model import swin_base_patch4_window7_224_in22k as swin_base

from vit_model import vit_base_patch16_224_in21k as vit_B
from vit_model import vit_large_patch16_224_in21k as vit_L
from peft import LoraConfig, get_peft_model, TaskType
torch.backends.cudnn.benchmark = True  # 提高 CNN 计算速度

os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"

# 加入存档，log

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lg', default=1.0, type=float, help='learning rate')
parser.add_argument('--epoch', default=1001, type=int, help='number of epochs to train')
parser.add_argument('--num_workers', default=100, type=int, help='#workers')
parser.add_argument('--batch_size', default=50, type=int, help='# batch_size')
parser.add_argument('--E', default=5, type=int, help='# batch_size')
parser.add_argument('--alg', default='FedAvg', type=str, help='FedAvg')  # FedMoment cddplus cdd SCAF atte
parser.add_argument('--extname', default='EM', type=str, help='extra_name')
parser.add_argument('--gpu', default='0', type=str, help='use which gpus')
parser.add_argument('--lr_decay', default='0.998', type=float, help='lr_decay')
parser.add_argument('--data_name', default='CIFAR100', type=str, help='imagenet,CIFAR100')
parser.add_argument('--lr_ps', default='1', type=float, help='only for FedAdam ')
parser.add_argument('--alpha_value', default='0.1', type=float, help='for dirichlet')
parser.add_argument('--selection', default='0.1', type=float, help=' C')
parser.add_argument('--check', default=0, type=int, help=' if check')
parser.add_argument('--alpha', default=0.01, type=float, help=' for mom_step')
parser.add_argument('--CNN', default='lenet5', type=str, help=' for mom_step')
parser.add_argument('--gamma', default=0.85, type=float, help=' for mom_step')
parser.add_argument('--p', default=10, type=float, help=' for mom_step')
parser.add_argument('--freeze-layers', type=bool, default=False)
parser.add_argument('--datapath', type=str, default="./data")
parser.add_argument('--T_part', default=10, type=int, help=' for mom_step')

parser.add_argument('--num_gpus_per', default=1, type=float, help=' for mom_step')
parser.add_argument('--normalization', default='BN', type=str, help=' for mom_step')
parser.add_argument('--pre', default=1, type=int, help=' for mom_step')
parser.add_argument('--print', default=0, type=int, help=' for mom_step')
parser.add_argument('--momentum', type=float, default=0.5, metavar='N', help='momentum')
parser.add_argument('--tau', default='0.01', type=float, help='only for FedAdam ')
# FedSAM
parser.add_argument("--rho", type=float, default=0.05, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--adaptive", type=bool, default=True, help="True if you want to use the Adaptive SAM.")
parser.add_argument("--preprint", type=int, default=10, help="")
parser.add_argument("--r", type=int, default=16, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
                    help='initial weights path')
parser.add_argument('--optimizer', type=str, default='SGD',
                    help='initial weights path')
parser.add_argument("--maxnorm", type=float, default=10, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--lora", type=int, default=1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--freeze', default=0, type=int, help='# batch_size')
parser.add_argument('--K', default=20, type=int, help='#workers')

args = parser.parse_args()
gpu_idx = args.gpu
print('gpu_idx', gpu_idx)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
num_gpus_per = args.num_gpus_per  # num_gpus_per = 0.16

num_gpus = len(gpu_idx.split(','))
# num_gpus_per = 1
data_name = args.data_name
CNN = args.CNN
print(CNN)

if CNN in ['VIT-B', 'swin_tiny', 'swin_large', 'VIT-L', 'swin_small', 'swin_base']:
    lora_config = LoraConfig(
        r=args.r,  # 低秩矩阵的秩，通常在 4 到 64 之间[^18^]
        lora_alpha=args.r,  # 缩放参数，通常为 r 的 2 到 32 倍[^18^]
        lora_dropout=0.05,  # Dropout 比率，防止过拟合[^18^]
        bias="none",  # 不训练偏置项[^18^]
        task_type="IMAGE_CLASSIFICATION",  # 任务类型，根据具体任务选择[^18^]
        target_modules=['attn.qkv', 'attn.proj']  # 目标模块，根据模型结构指定[^18^]
    )



if CNN in ['VIT-B', 'swin_tiny', 'swin_large', 'VIT-L', 'swin_small', 'swin_base', 'resnet18pre', 'resnet50pre',
           'resnet101pre']:
    if data_name == 'imagenet':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),  # 将图像大小调整为 ResNet-18 输入的大小
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])

    if data_name == 'CIFAR100' or data_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

else:
    if data_name == 'CIFAR10' or data_name == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )

    if data_name == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
        ])

import dataset as local_datasets

if data_name == 'imagenet':
    train_dataset = local_datasets.TinyImageNetDataset(
        root=os.path.join(args.datapath, 'tiny-imagenet-200'),
        split='train',
        transform=transform_train
    )

if data_name == 'CIFAR10':

    train_dataset = datasets.CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transform_train)


elif data_name == 'CIFAR100':
    train_dataset = datasets.cifar.CIFAR100(
        "./data",
        train=True,
        download=True,
        transform=transform_train
    )
seed=42
if args.alpha_value==1:
    generator = torch.Generator().manual_seed(42)
    total_size = len(train_dataset)
    print(total_size)
    subset_size = total_size // args.num_workers
    remainder = total_size % args.num_workers  # 计算剩余的样本数
    # 创建分割大小列表
    split_sizes = [subset_size] * (args.num_workers-1)+ [subset_size + remainder]
    subsets = random_split(train_dataset, split_sizes, generator=generator)

    def get_data_loader(pid, data_idx, batch_size, data_name):
        """Safely downloads data. Returns training/validation set dataloader. 使用到了外部的数据"""
        sample_chosed = data_idx[pid]
        train_sampler = SubsetRandomSampler(sample_chosed)
        train_loader = DataLoader(subsets[pid], batch_size=args.batch_size, shuffle=True)
        return train_loader

if args.alpha_value!=1:
    def get_data_loader(pid, data_idx, batch_size, data_name):
        """Safely downloads data. Returns training/validation set dataloader. 使用到了外部的数据"""
        sample_chosed = data_idx[pid]
        train_sampler = SubsetRandomSampler(sample_chosed)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler, num_workers=0, generator=torch.Generator().manual_seed(seed))
        return train_loader


def get_data_loader_test(data_name):
    """Safely downloads data. Returns training/validation set dataloader."""

    if data_name == 'imagenet':
        test_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join(args.datapath, 'tiny-imagenet-200'),
            split='test',
            transform=transform_train
        )
    if data_name == 'CIFAR10':
        test_dataset = datasets.CIFAR10("./data", train=False, transform=transform_train)

    elif data_name == 'CIFAR100':
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, transform=transform_train
                                               )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=4)
    return test_loader


def get_data_loader_train(data_name):
    """Safely downloads data. Returns training/validation set dataloader."""
    if data_name == 'imagenet':
        train_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join(args.datapath, 'tiny-imagenet-200'),
            split='train',
            transform=transform_train
        )
    if data_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10("./data", train=True, transform=transform_train)
        # test_dataset = datasets.cifar.CIFAR100("./data", train=False, transform=transform_test)

    elif data_name == 'CIFAR100':
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, transform=transform_train
                                                )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=4)
    return train_loader


if data_name == 'imagenet' or data_name == 'CIFAR10' or data_name == 'CIFAR100':
    def evaluate(model, test_loader, train_loader):
        """Evaluates the accuracy of the model on a validation dataset."""
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        model = model.to(device)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        torch.cuda.empty_cache()
        return 100. * correct / total, torch.tensor(0), torch.tensor(0)




if CNN == 'swin_tiny':
    def ConvNet():
        return swin_tiny(num_classes=10)


    def ConvNet100():
        return swin_tiny(num_classes=100)


    def ConvNet200():
        return swin_tiny(num_classes=200)

if CNN == 'swin_large':
    def ConvNet():
        return swin_large(num_classes=10)


    def ConvNet100():
        return swin_large(num_classes=100)


    def ConvNet200():
        return swin_large(num_classes=200)
if CNN == 'swin_small':
    def ConvNet():
        return swin_small(num_classes=10)


    def ConvNet100():
        return swin_small(num_classes=100)


    def ConvNet200():
        return swin_small(num_classes=200)

if CNN == 'swin_base':
    def ConvNet():
        return swin_base(num_classes=10)


    def ConvNet100():
        return swin_base(num_classes=100)


    def ConvNet200():
        return swin_base(num_classes=200)

if CNN == 'VIT-B':
    def ConvNet():
        return vit_B(num_classes=10)


    def ConvNet100():
        return vit_B(num_classes=100)


    def ConvNet200():
        return vit_B(num_classes=200)
if CNN == 'VIT-L':
    def ConvNet():
        return vit_L(num_classes=10)


    def ConvNet100():
        return vit_L(num_classes=100)


    def ConvNet200():
        return vit_L(num_classes=200)


import torch
import torch.nn as nn




@ray.remote
# @ray.remote(num_gpus=args.num_gpus_per)
class ParameterServer(object):
    def __init__(self, lr, alg, tau, selection, data_name, num_workers):
        if data_name == 'CIFAR10':
            self.model = ConvNet()
        elif data_name == 'CIFAR100':
            self.model = ConvNet100()
        if data_name == 'imagenet':
            self.model = ConvNet200()
        if data_name == 'FashionMNIST':
            self.model = ConvNet()
        if data_name == 'SVHN':
            self.model = ConvNet()
        if data_name == 'EMNIST':
            self.model = ConvNet()
        if data_name == 'MNIST':
            self.model = ConvNet()
        # self.model = torch.compile(self.model)
        if args.lora == 1:
            self.model = get_peft_model(self.model, lora_config)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        # self.momen_v = None
        # self.gamma = 0.9
        self.gamma = args.gamma
        # self.gamma = 0.9
        self.beta = 0.99  # 论文是0.99
        self.alg = alg
        self.num_workers = num_workers

        self.lr_ps = lr
        self.lg = 1.0
        self.ps_c = None
        self.c_all = None
        # 上一代的c
        self.c_all_pre = None
        self.tau = tau
        self.selection = selection
        self.cnt = 0
        self.alpha = args.alpha
        self.h = {}
        self.momen_m = {}
        self.momen_v = {}

    def set_pre_c(self, c):
        self.c_all_pre = c

    def apply_weights_avg(self, num_workers, *weights):

        ps_w = self.model.get_weights()  # w : ps_w
        print(ps_w.keys())
        sum_weights = {}  # delta_w : sum_weights
        global_weights = {}
        for weight in weights:
            for k, v in weight.items():
                if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                    sum_weights[k] += v / (num_workers * self.selection)
                else:

                    sum_weights[k] = v / (num_workers * self.selection)
        for k, v in sum_weights.items():  # w = w + delta_w
            global_weights[k] = ps_w[k] + sum_weights[k]
        self.model.load_state_dict(global_weights)
        return self.model.get_weights()





    def load_dict(self):
        self.func_dict = {
            'FedLORA': self.apply_weights_avg,
            'SAM-LORA': self.apply_weights_avg,
            'FFA-LORA':self.apply_weights_avg,
            'AR-LORA':self.apply_weights_avg,

        }

    def apply_weights_func(self, alg, num_workers, *weights):
        self.load_dict()
        return self.func_dict.get(alg, None)(num_workers, *weights)



    def get_weights(self):
        return self.model.state_dict()

    def get_ps_c(self):
        return self.ps_c

    def get_state(self):
        return self.ps_c, self.c_all

    def set_state(self, c_tuple):
        self.ps_c = c_tuple[0]
        self.c_all = c_tuple[1]

    def set_weights(self, weights):
        self.model.load_state_dict(weights)



@ray.remote(num_gpus=num_gpus_per)
class DataWorker(object):

    def __init__(self, pid, data_idx, num_workers, lr, batch_size, alg, data_name, selection, T_part):
        self.alg = alg
        if data_name == 'CIFAR10':
            self.model = ConvNet().to(device)
        elif data_name == 'CIFAR100':
            self.model = ConvNet100().to(device)
        if data_name == 'imagenet':
            self.model = ConvNet200().to(device)
        if data_name == 'FashionMNIST':
            self.model = ConvNet().to(device)
        if data_name == 'SVHN':
            self.model = ConvNet().to(device)
        if data_name == 'EMNIST':
            self.model = ConvNet().to(device)
        if data_name == 'MNIST':
            self.model = ConvNet().to(device)
        if args.lora == 1:
            self.model = get_peft_model(self.model, lora_config)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.pid = pid
        self.num_workers = num_workers
        self.data_iterator = None
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.loss = 0
        self.lr_decay = lr_decay
        self.alg = alg
        self.data_idx = data_idx
        self.pre_ps_weight = None
        self.pre_loc_weight = None
        self.flag = False
        self.ci = None
        self.selection = selection
        self.T_part = T_part
        self.Li = None
        self.hi = None
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.dp_clip = 10
        self.R=1

    def data_id_loader(self, index):
        '''
        在每轮的开始，该工人装载数据集，以充当被激活的第index个客户端
        '''
        self.data_iterator = get_data_loader(index, self.data_idx, batch_size, data_name)

    def state_id_loader(self, index):
        '''
        在每轮的开始，该工人装载状态，以充当被激活的第index个客户端，使用外部的状态字典
        '''
        if not c_dict.get(index):
            return
        self.ci = c_dict[index]

    def state_hi_loader(self, index):
        if not hi_dict.get(index):
            return
        self.hi = hi_dict[index]

    def state_Li_loader(self, index):
        if not Li_dict.get(index):
            return
        self.Li = Li_dict[index]

    def get_train_loss(self):
        return self.loss





    def update_FedAvg_per(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if args.freeze==0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=1e-3)
        if args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        step = 0  # 新增步数计数
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                data = data.to(device)
                target = target.to(device)
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                step += 1
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        return delta_w

    def update_SAM(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, weight_decay=0.001, rho=args.rho)
        step = 0  # 新增步数计数
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step=step+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                self.criterion(self.model(data), target).backward()
                self.optimizer.second_step(zero_grad=True)
        self.loss = loss.item()
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        return delta_w

    def update_FedAvg_AL(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if args.freeze==0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True
        self.optimizer = torch.optim.SGD([
                {"params": [p for n, p in self.model.named_parameters() if "lora_A" in n], "lr": lr,"weight_decay":  0.001},
                {"params": [p for n, p in self.model.named_parameters() if "lora_B" in n], "lr": lr * 2,"weight_decay":  0.001},
                {"params": [p for n, p in self.model.named_parameters() if "lora_" not in n], "lr": lr,"weight_decay": 0.001}
            ])
        if args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW([
                {"params": [p for n, p in self.model.named_parameters() if "lora_A" in n], "lr": lr},
                {"params": [p for n, p in self.model.named_parameters() if "lora_B" in n], "lr": lr * 2},
                {"params": [p for n, p in self.model.named_parameters() if "lora_" not in n], "lr": lr}
            ], weight_decay=0.01)
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        self.laplacian = args.laplacian
        self.ls_sigma = args.ls_sigma
        step = 0  # 新增步数计数
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                for name, param in self.model.named_parameters():
                    if batch_idx%2==1:
                        if "lora_A" in name:
                            param.requires_grad = False
                        if "lora_B" in name:
                            param.requires_grad = True
                    if batch_idx%2==0:
                        if "lora_A" in name:
                            param.requires_grad = True
                        if "lora_B" in name:
                            param.requires_grad = False
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                step += 1
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        return delta_w

    def update_FFA_LoRA(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        for name, param in self.model.named_parameters():
            if 'lora_A' in name:
                param.requires_grad = False
        if args.freeze==0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=1e-3)
        if args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        step = 0  # 新增步数计数
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target.long())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                step += 1
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        return delta_w

    def update_AR_LoRA(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        for name, param in self.model.named_parameters():
            if self.R%2==1:
                if 'lora_A' in name:
                    param.requires_grad = False
                if 'lora_B' in name:
                    param.requires_grad = True
            if self.R%2==0:
                if 'lora_B' in name:
                    param.requires_grad = False
                if 'lora_A' in name:
                    param.requires_grad = True
        if args.freeze==0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                         weight_decay=1e-3)
        if args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                               weight_decay=0.01)
        self.data_id_loader(index)
        step = 0  # 新增步数计数
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target.long())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                step += 1
        self.R=self.R+1
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        return delta_w








    def load_dict(self):
        self.func_dict = {

            'FedLORA': self.update_FedAvg_per,
            'FFA-LORA': self.update_FFA_LoRA,
            'AR-LORA': self.update_AR_LoRA,
            'SAM-LORA': self.update_SAM,

        }

    def update_func(self, alg, weights, E, index, lr, ps_c=None):
        self.load_dict()
        if alg in {'DP-SCAFFOLD',}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, lr)
        else:
            return self.func_dict.get(alg, None)(weights, E, index, lr)

    import random
    import numpy as np
    import torch


def set_random_seed(seed=42):
    """
    设置随机种子以确保实验的可重复性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def apply_weights_avg(num_workers, weights,model):

    ps_w = model.state_dict()  # w : ps_w
    #print(ps_w.keys())
    sum_weights = {}  # delta_w : sum_weights
    global_weights = {}
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)
    for k, v in sum_weights.items():  # w = w + delta_w
        global_weights[k] = ps_w[k] + sum_weights[k]
    model.load_state_dict(global_weights)
    return model.state_dict()




if __name__ == "__main__":
    # 获取args
    set_random_seed(seed=42)
    epoch = args.epoch
    num_workers = args.num_workers
    batch_size = args.batch_size
    lr = args.lr
    E = args.E
    lr_decay = args.lr_decay  # for CIFAR10
    # lr_decay = 1
    alg = args.alg
    data_name = args.data_name
    selection = args.selection
    tau = args.tau
    lr_ps = args.lr_ps
    alpha_value = args.alpha_value
    alpha = args.alpha
    extra_name = args.extname
    check = args.check
    T_part = args.T_part
    c_dict = {}
    lr_decay = args.lr_decay

    hi_dict = {}
    Li_dict = {}
    import time

    localtime = time.asctime(time.localtime(time.time()))

    checkpoint_path = './checkpoint/ckpt-{}-{}-{}-{}-{}-{}'.format(alg, lr, extra_name, alpha_value, extra_name,
                                                                   localtime)
    c_dict = {}  # state dict
    assert alg in {
        'FedLORA',
        'FFA-LORA',
        'AR-LORA',
        'SAM-LORA',


    }
    #  配置logger
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("./log/{}-{}-{}-{}-{}-{}-{}.txt"
                                  .format(alg, data_name, lr, num_workers, batch_size, E, lr_decay))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    writer = SummaryWriter(comment=alg)

    nums_cls = 100
    if data_name == 'CIFAR10':
        nums_cls = 10
    if data_name == 'CIFAR100':
        nums_cls = 100
    if data_name == 'imagenet':
        nums_cls = 200
    if data_name == 'FashionMNIST':
        nums_cls = 10
    if data_name == 'SVHN':
        nums_cls = 10
    if data_name == 'EMNIST':
        nums_cls = 47
    if data_name == 'MNIST':
        nums_cls = 10

    nums_sample = 500
    if data_name == 'CIFAR10':
        nums_sample = int(50000 / (args.num_workers))
        nums_sample = 500
    if data_name == 'CIFAR100':
        nums_sample = int(50000 / (args.num_workers))
        nums_sample = 500
    if data_name == 'imagenet':
        nums_sample = int(100000 / (args.num_workers))
    if data_name == 'FashionMNIST':
        nums_sample = int(60000 / (args.num_workers))
    if data_name == 'SVHN':
        nums_sample = int(70000 / (args.num_workers))
    if data_name == 'EMNIST':
        nums_sample = 500
    if data_name == 'MNIST':
        nums_sample = 500

    import pickle

    if args.data_name == 'imagenet':
        # 存储变量的文件的名字
        if args.alpha_value == 0.6:
            filename = 'data_idx.data'
        if args.alpha_value == 0.1:
            filename = 'data_idx100000_0.1.data'
        f = open(filename, 'rb')
        # 将文件中的变量加载到当前工作区
        data_idx = pickle.load(f)
    else:
        data_idx, std = data_from_dirichlet(data_name, alpha_value, nums_cls, num_workers, nums_sample)
        logger.info('std:{}'.format(std))
    #
    ray.init(ignore_reinit_error=True, num_gpus=num_gpus)

    #ps = ParameterServer.remote(lr_ps, alg, tau, selection, data_name, num_workers)
    if data_name == 'imagenet':
        model = ConvNet200().to(device)
    if data_name == 'CIFAR10':
        model = ConvNet().to(device)
    elif data_name == 'CIFAR100':
        model = ConvNet100().to(device)
    if data_name == 'FashionMNIST':
        model = ConvNet().to(device)
    if data_name == 'SVHN':
        model = ConvNet().to(device)
    if data_name == 'EMNIST':
        model = ConvNet().to(device)
    if data_name == 'MNIST':
        model = ConvNet().to(device)

    epoch_s = 0
    # c_dict = None,None
    workers = [DataWorker.remote(i, data_idx, num_workers,
                                 lr, batch_size=batch_size, alg=alg, data_name=data_name, selection=selection,
                                 T_part=T_part) for i in range(int(num_workers * selection / args.p))]
    logger.info(
        'extra_name:{},alg:{},E:{},data_name:{}, epoch:{}, lr:{},alpha_value:{},alpha:{},CNN:{},rho:{},C:{},sigma:{},name:{}'
        .format(extra_name, alg, E, data_name, epoch, lr, alpha_value, alpha, args.CNN, args.rho, args.C, args.dp_sigma,
                args.alg))
    # logger.info('data_idx{}'.format(data_idx))
    test_loader = get_data_loader_test(data_name)
    train_loader = get_data_loader_train(data_name)
    print("@@@@@ Running synchronous parameter server training @@@@@@")

    if args.CNN == 'VIT-B':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('vit_base_patch16_224_in21k.pth', map_location=device)
            # 删除不需要的权重
            del_keys = ['head.weight', 'head.bias'] if model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head, pre_logits外，其他权重全部冻结
                if "head" not in name and "pre_logits" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

    if args.CNN == 'VIT-L':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('jx_vit_large_patch16_224_in21k-606da67d.pth', map_location=device)
            # 删除不需要的权重
            del_keys = ['head.weight', 'head.bias'] if model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head, pre_logits外，其他权重全部冻结
                if "head" not in name and "pre_logits" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

    if args.CNN == 'swin_tiny':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_tiny_patch4_window7_224.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))


    if args.CNN == 'swin_small':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_small_patch4_window7_224.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

    if args.CNN == 'swin_base':

        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_base_patch4_window7_224_22k.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))
        # '''

    if args.CNN == 'swin_large':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_large_patch4_window7_224_22k.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))


    if CNN in ['VIT-B', 'swin_tiny', 'swin_large', 'VIT-L', 'swin_small', 'swin_base']:
        lora_config = LoraConfig(
            r=args.r,  # 低秩矩阵的秩，通常在 4 到 64 之间[^18^]
            lora_alpha=args.r,  # 缩放参数，通常为 r 的 2 到 32 倍[^18^]
            lora_dropout=0.05,  # Dropout 比率，防止过拟合[^18^]
            bias="none",  # 不训练偏置项[^18^]
            task_type="IMAGE_CLASSIFICATION",  # 任务类型，根据具体任务选择[^18^]
            target_modules=['attn.qkv', 'attn.proj']  # 目标模块，根据模型结构指定[^18^]
        )
    if args.lora == 1:
        model = get_peft_model(model, lora_config)
    model=model.to('cpu')

    #ps.set_weights.remote(model.state_dict())
    current_weights = model.state_dict()
    #ps_c = ps.get_ps_c.remote()

    result_list, X_list = [], []
    result_list_loss = []
    test_list_loss = []
    start = time.time()
    # for early stop
    best_acc = 0
    no_improve = 0
    zero = model.get_weights()
    # print(delta_g_sum)
    for k, v in model.get_weights().items():
        zero[k] = zero[k] - zero[k]
    ps_c = deepcopy(zero)

    del zero
    for epochidx in range(epoch_s, epoch):
        start_time1 = time.time()
        index = np.arange(num_workers)  # 100
        lr = lr * lr_decay
        np.random.shuffle(index)

        index = index[:int(num_workers * selection)]  # 10id


        if alg in {'FedLORA','FFA-LORA','AR-LORA','SAM-LORA'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                # worker_sel = workers[i:i + int(n / 2)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                     zip(workers, index_sel)]
            weights=ray.get(weights)
            model.to('cpu')
            #current_weights =apply_weights_avg_LS(num_workers, weights,model)
            current_weights =apply_weights_avg(num_workers, weights,model)
            #current_weights = ps.apply_weights_func.remote(alg, num_workers, *weights)
            #current_weights = ray.get(current_weights)
            model.load_state_dict(current_weights)

        end_time1 = time.time()
        #print(epochidx, '    ', end_time1 - time3)
        print(epochidx, '    ', end_time1 - start_time1)

        if epochidx % args.preprint == 0:
            start_time1 = time.time()
            print('测试')
            test_loss = 0
            train_loss = 0
            model.load_state_dict(current_weights)
            accuracy, test_loss, train_loss = evaluate(model, test_loader, train_loader)
            end_time1 = time.time()
            print('测试完毕', '    ', end_time1 - start_time1)
            test_loss = test_loss.to('cpu')
            loss_train_median = train_loss.to('cpu')
            # early stop
            if accuracy > best_acc:
                best_acc = accuracy
                #ps_state = ps.get_state.remote()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve == 1000:
                    break

            writer.add_scalar('accuracy', accuracy, epochidx * E)
            writer.add_scalar('loss median', loss_train_median, epochidx * E)
            logger.info(
                "Iter {}: \t accuracy is {:.1f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}, name:{},C:{},sigma:{},lr:{:.5f},CNN:{},GPU:{},gamma:{},rho:{},alpha_value:{},ls_sigma:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    no_improve, args.alg, args.C, args.dp_sigma, lr, args.CNN, args.gpu, args.gamma, args.rho,
                    args.alpha_value, args.ls_sigma))

            #print(
            #    "Iter {}: \t accuracy is {:.1f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}, name:{},C:{},sigma:{},lr:{:.5f},CNN:{},GPU:{},data:{},gamma:{},rho:{},alpha_value:{}".format(
            #        epochidx, accuracy,
            #        loss_train_median, test_loss,
            #        no_improve, args.alg, args.C, args.dp_sigma, lr, args.CNN, args.gpu, args.data_name, args.gamma,
            #        args.rho, args.alpha_value))

            # logger.info('attention:{}'.format(ray.get(ps.get_attention.remote())))
            if np.isnan(loss_train_median):
                logger.info('nan~~')
                break
            X_list.append(epochidx)
            result_list.append(accuracy)
            result_list_loss.append(loss_train_median)
            test_list_loss.append(test_loss)

    logger.info("Final accuracy is {:.2f}.".format(accuracy))
    endtime = time.time()
    logger.info('time is pass:{}'.format(endtime - start))
    x = np.array(X_list)
    result = np.array(result_list)

    result_loss = np.array(result_list_loss)
    test_list_loss = np.array(test_list_loss)
    # x2 = np.array(X_list)
    # div = np.array(div)

    save_name = './plot/alg_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-rho{}-CNN{}-time{}-C{}-sigma{}-ls_sigma{}'.format(
        alg, E, num_workers, epoch,
        lr, alpha_value, selection, alpha,
        args.data_name, args.gamma, args.rho, args.CNN, endtime, args.C, args.dp_sigma, args.ls_sigma)

    save_name2 = './model/model_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-rho{}-CNN{}-time{}-C{}-sigma{}'.format(
        alg, E, num_workers, epoch,
        lr, alpha_value, selection, alpha,
        args.data_name, args.gamma, args.rho, args.CNN, endtime, args.C, args.dp_sigma)
    torch.save(model.state_dict(), save_name2)
    save_name = save_name + '.npy'
    # save_name2 = save_name2 + '.pth'
    np.save(save_name, (x, result, result_loss, test_list_loss))
    ray.shutdown()