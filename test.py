## Here you may implement the evaluation method and may call some necessary modules from utils.model_utils.py
## Derive the test function by yourself and implement proper metric such as Dice similarity coeffcient (DSC)[4];
# Jaccard distance[5] and Mean squared error (MSE), etc. following the handout in model_utilss.py

import numpy as np
import torch
from tqdm import tqdm
import os
from torch import optim
from torch.utils import data
from torch import nn
from torch.utils.data import DataLoader
from utils.FragmentDataset import FragmentDataset
from torchvision import datasets, transforms
from utils.model import Generator, Discriminator
from utils.model_utils import posprocessing, generate, DSC, JD, MSE
import click
import argparse

def test(G, epoch):
    Z_latent_space = 64
    loss_function = 'BCE'
    generator_learning_rate = 0.002
    discriminator_learning_rate = 0.0002
    initial_data_resolution = 32
    optimizer = 'ADAM'
    beta1 = 0.9
    beta2 = 0.999
    batch_size = 64
    available_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_dir = "./data"
    result_save_dir = "./result"

    # TODO
    # You can also implement this function in training procedure, but be sure to
    # evaluate the model on test set and reserve the option to save both quantitative
    # and qualitative (generated .vox or visualizations) images. 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G.to(device)
    checkpoint = torch.load(os.path.join(result_save_dir, f"generator_epoch_{epoch}.pth"))
    G.load_state_dict(checkpoint)
    test_dataset = FragmentDataset(dataset_dir, 'vox', resolution=initial_data_resolution, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    G.eval()
    total = 0
    test_DSC = 0.0
    test_JD = 0.0
    test_MSE = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            real_frag = data[0].to(available_device).float()
            real_vox_wo_frag = data[1].to(available_device).float()
            real_vox = (data[0] + data[1]).to(available_device).float()
            real_frag = real_frag.unsqueeze(1)
            real_vox_wo_frag = real_vox_wo_frag.unsqueeze(1)
            real_vox = real_vox.unsqueeze(1)
            #print(real_frag.shape)
            
            real_frag = real_frag.cpu().detach().numpy()
            fake, mesh_frag = generate(G, real_vox_wo_frag)
            #print(fake.shape)
            test_DSC += DSC(fake, real_frag)
            test_JD += JD(fake, real_frag)
            test_MSE += MSE(fake, real_frag)
            total += 1
            
        test_DSC /= total
        test_JD /= total
        test_MSE /= total
        print(f'Test Avg DSC: {test_DSC}')
        print(f'Test Avg JD: {test_JD}')
        print(f'Test Avg MSE: {test_MSE}')

    return test_DSC, test_JD, test_MSE