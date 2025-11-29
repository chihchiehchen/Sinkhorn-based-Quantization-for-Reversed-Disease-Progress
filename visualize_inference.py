
from optvq.utils.init import initiate_from_config_recursively
from torch.utils.data.dataloader import DataLoader

from optvq.trainer.pipeline import (
    get_pipeline, get_setup_optimizers, 
    setup_config, setup_dataset, 
    setup_dataloader, setup_model
)
import optvq.utils.logger as L

from timm import create_model
import copy
from omegaconf import DictConfig, OmegaConf

from accelerate.utils import set_seed
import argparse
import os
from functools import partial
import torch
from utils import infiniteloop, MLS_Taming, encode_vae, decode_vae, generate_latent_samples_single_case_with_patch 

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper
from torchvision.utils import save_image

from tqdm import trange



def warmup_lr(step):
    return min(step, 5000) / 5000



def main(configs, checkpoint_path=None, latent_path = None, png_dir= None, output_dir = None):
    copy_of_configs = copy.deepcopy(configs)
    set_seed(configs.pop("seed", 42))
    
    L.config_loggers(log_dir=output_dir, local_rank=gpu)
    L.log.info(f"Start a new training logger at {output_dir}")

    vae_model = initiate_from_config_recursively(configs.model.autoencoder)

    print(f" *** Model: {vae_model} ***")
    print(
        f" *** Parameters: {sum(p.numel() for p in vae_model.parameters())/1e6:.2f}M ***"
    )

    # load checkpoint
    pkg = torch.load(str(checkpoint_path))
    
    revised = {}
    for key in pkg["model"]:
        r_key = key.replace('module.',"")
        revised[r_key] = pkg['model'][key]

    vae_model.load_state_dict(revised)
    vae_model.cuda()
    vae_model.eval()

     
    eval_data = MLS_Taming(root=png_dir)

    
    train_loader = DataLoader(
                dataset=train_data, batch_size=configs.data.batch_size, num_workers=8,
                drop_last=True, shuffle=True, persistent_workers=True, pin_memory=True
            )

    

    # build eval data loader
    
    eval_loader = DataLoader(
                dataset=eval_data, batch_size=80, num_workers=8,
                drop_last=False, shuffle=False, persistent_workers=True, pin_memory=True
            )



    print(f" *** Eval data loader: {len(eval_loader)} ***")

    looper_Val = infiniteloop(eval_loader)

    net_model = UNetModelWrapper(
        dim=(256, 16, 16),
        num_res_blocks=2,
        num_channels=128,
        num_heads=4,
        channel_mult = (1,2, 2,4),
        attention_resolutions="16",
        dropout=0.1,
    ).cuda()


    latent_checkpoint = torch.load(args.latent_path)['net_model'] 
    net_model.load_state_dict(latent_checkpoint)
    net_model.eval()


    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    fm_method = configs.pop('fm' , 'otcfm')

    savedir = os.path.join(output_dir , checkpoint_path.split('/')[-1].replace('.pt',""),fm_method)
    os.makedirs(savedir, exist_ok=True)

    
    if parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)

    for i, (x0_input, path_code) in enumerate(eval_loader):
        x0_input = x0_input.cuda()            
        l_size = x0_input.size()[0]
        with torch.no_grad():
             x0 = encode_vae(vae_model, x0_input,divide=False)
             generate_latent_samples_single_case_with_patch(vae_model,net_model, parallel, savedir, i, x0, path_code=path_code, net_=args.net_type,divide=False, use_quant=False)
             save_image(torch.clamp(0.5*x0_input+0.5,0,1), os.path.join(savedir , f"val_pixel_orig_step_{i}.png"), nrow=8)

    
def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("F-MAE training", add_help=add_help)
    parser.add_argument(
        "--config-file",
        "--config_file",
        default="./configs/mls/latent_optvq_class_next_256_f16_h4.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--latent_path",
        "--latent-path",
        help="manually restore from a specific latent checkpoint directory",
    )

    parser.add_argument(
        "--net_type",
        "--net-type",
        default="net_model",
        help="net_model or ema_model",
    )

    parser.add_argument(
        "--checkpoint_path",
        "--checkpoint-path",
        help="manually restore from a specific checkpoint directory",
    )
    
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="logs_TEST",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    
    parser.add_argument(
        "--png-dir",
        "--png_dir",
        default="logs_png",
        type=str,
        help="image directory",
    )


    return parser


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    cfg = OmegaConf.load(args.config_file)
    main(cfg, checkpoint_path=args.checkpoint_path, latent_path = args.latent_path ,  png_dir = args.png_dir, output_dir = args.output_dir)
