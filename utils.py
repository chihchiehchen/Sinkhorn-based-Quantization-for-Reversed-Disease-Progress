import copy, os 

import torch
from torch.utils.data import Dataset

from torchdyn.core import NeuralODE

from torchvision.utils import make_grid, save_image
import numpy as np
from PIL import Image


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class MLS_Taming(Dataset):
    def __init__(self, root, transform=None, convert_to_numpy: bool = True, post_normalize: str = "plain"):
        self.root = root
        self.transform = transform
        self.convert_to_numpy = convert_to_numpy
        self.post_normalize = transforms.Normalize(
            **normalize_params[post_normalize]
        )

        self.samples = [os.path.join(self.root, x) for x in os.listdir(self.root)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((256,256))
        if self.convert_to_numpy:
            image = np.array(image).astype("uint8")
        
        image = (image / 255).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = self.post_normalize(image)
        
        path_code = path.split('/')[-1].split('.')[0]

        return image, path_code



def encode_vae(vae_model, x, divide=False):
    l_size = x.size()[0]
    with torch.no_grad():
        if divide:
            x1, _ , _ = vae_model.encode(x[0:l_size//3])
            x2, _ , _ = vae_model.encode(x[l_size//3:2*l_size//3])
            x3, _ , _ = vae_model.encode(x[2*l_size//3: -1])
            x = torch.cat((x1,x2,x3), dim =0 )
        else:
            x, _ , _ = vae_model.encode(x)
    return x

def decode_vae(vae_model, x, divide=False):
    l_size = x.size()[0]
    with torch.no_grad():
        if divide:
            x1 = vae_model.decode(x[0:l_size//3])     
            x2 = vae_model.decode(x[l_size//3:2*l_size//3])             
            x3 = vae_model.decode(x[2*l_size//3: -1])             
            x = torch.cat((x1,x2,x3), dim =0 )           
        else:
            x = vae_model.decode(x)
    return x



def infiniteloop(dataloader):
    while True: 
        for (x, path_code) in iter(dataloader):
            yield x, path_code


def generate_latent_samples_single_case_with_patch(vae_model,model, parallel, savedir, step, x0, path_code ,net_="normal",divide=False, num_pic =10, use_patch= True, use_quant= True):
    """Save 64 generated images (8 x 8) for sanity check along training.
    
    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images    
    step: int
        represents the current step of training
    """
    model.eval()
    _, _, h, w = x0.size()
    
    mask = torch.zeros((h,w))

    mask[4:8, 6:10] = 1

    mask = mask[None, None, :, :].to(x0.device)

    model_ = copy.deepcopy(model)

    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)
            
    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            x0,
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        l_size= x0.size()[0]
        for i in range(l_size):
          
            if vae_model != None:
                if use_patch:
                    curr_traj_i = ((1- mask)*x0)[i].unsqueeze(0) + mask*traj[:,i,:,:,:]
                else:
                    curr_traj_i = traj[:,i,:,:,:]

                t_list = []
                q_list = []
                traj_quant, _, _ = vae_model.quantize(curr_traj_i[:50,:,:,:])
                q_list.append(decode_vae(vae_model,traj_quant))
                traj_i = decode_vae(vae_model, curr_traj_i[:50,:,:,:])

                t_list.append(traj_i)
                if use_quant:
                    traj_quant, _, _ = vae_model.quantize(curr_traj_i[50:,:,:,:])
                    q_list.append(decode_vae(vae_model, traj_quant))

                traj_i = decode_vae(vae_model,curr_traj_i[50:,:,:,:])
                t_list.append(traj_i)
                traj_i = torch.cat(t_list, dim=0)
                if use_quant:
                    traj_i_quant =  torch.cat(q_list, dim=0)

            else:
                if use_patch:
                    traj_i = ((1- mask)*x0)[i].unsqueeze(0) + mask*traj[:,i,:,:,:]
                else:
                    traj_i = traj[:,i,:,:,:]

            traj_i = traj_i / 2 + 0.5
            traj_i = torch.clamp(traj_i,0,1)
            if use_quant:
                traj_i_quant = traj_i_quant / 2 + 0.5
                traj_i_quant = torch.clamp(traj_i_quant,0,1)
           
            total = traj_i.size()[0]
            for j in range(total):
                if j % 20 == 0 or j == total -1:
                    temp = torch.unsqueeze(traj_i[j], 0)
                    save_image(temp, os.path.join(savedir , f"{net_}_{path_code[i]}_{j}.png"), nrow=1)
                    if use_quant:

                        temp = torch.unsqueeze(traj_i_quant[j], 0)
                        save_image(temp, os.path.join(savedir , f"{net_}_{path_code[i]}_quant_{j}.png"), nrow=1)          

        x0 = decode_vae(vae_model, x0,divide=divide)        
        x0 = x0 / 2 + 0.5
        x0 = torch.clamp(x0,0,1)

    save_image(x0, os.path.join(savedir , f"{net_.split('_')[0]}_orig_step_{step}.png"), nrow=8)
    model.train()

