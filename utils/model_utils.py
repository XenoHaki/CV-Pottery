import numpy as np
import torch
from utils.model import Generator, Discriminator
from utils.visualize import *

# Try to implement proper metric for test function
def DSC(prediction, target):
    prediction = prediction.flatten()
    target = target.flatten()

    intersection = np.sum(prediction * target)

    prediction_sum = np.sum(prediction)
    target_sum = np.sum(target)

    dice = (2.0 * intersection) / (prediction_sum + target_sum)
    return dice

def JD(prediction, target):
    prediction = prediction.flatten()
    target = target.flatten()

    intersection = np.sum(prediction * target)
    union = np.sum(np.logical_or(prediction, target))

    jaccard = 1 - (intersection / union)
    return jaccard

def MSE(prediction, target):
    prediction = prediction.flatten()
    target = target.flatten()

    mse = np.mean((prediction - target) ** 2)
    return mse

# Try to implement some post-processing methods for visual evaluation, Display your generated results
def posprocessing(fake, mesh_frag):
    # fake is the generated M*M*(1 or 4) output, try to recover a voxel from it 
    # design by yourself or you can also choose to ignore this function
    if fake.shape[2] == 1:
        threshold = 0.5
        voxel = np.where(fake >= threshold, 1, 0)
    elif fake.shape[2] == 4:
        voxel = np.argmax(fake, axis=2)
    else:
        print("Unsupported shape for fake")
    return voxel


# You can implement the below two functions to load checkpoints and visualize .vox files. Option choice

available_device = 'cuda'

def load_generator(path_checkpoint):
    ## for evaluation?
    G_encode_decode = Generator().to(available_device) #  hyperparams need to be implemented
    checkpoint = torch.load(path_checkpoint, map_location=available_device)
    G_encode_decode.load_state_dict(checkpoint)
    G_encode_decode = G_encode_decode.eval()

    return G_encode_decode


def generate(model, vox_frag):
    '''
    generate model, doesn't guaruantee 100% correct
    '''
    mesh_frag = torch.Tensor(vox_frag).float().to(available_device)
    #print(mesh_frag.shape)
    output_g_encode = model.forward_encode(mesh_frag)
    fake = model.forward_decode(output_g_encode)
    fake = fake + mesh_frag
    fake = fake.detach().cpu().numpy()
    mesh_frag = mesh_frag.detach().cpu().numpy()
    return fake, mesh_frag