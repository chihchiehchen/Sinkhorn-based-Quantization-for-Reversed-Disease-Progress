import json
from tqdm import tqdm
import argparse
import os
import kornia as K
import kornia.geometry as KG
import cv2 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pickle
def get_args_parser():
    parser = argparse.ArgumentParser('args script', add_help=False)

    
    
    parser.add_argument('--source-img', default='./register_img/CQ500CT432_25.png', type=str, help='source_dir')
    parser.add_argument('--mls-txt', default='/home/chihchieh/projects/taming_mls/shift_stat.pkl', type=str, help='json_dir')
    parser.add_argument('--orig-dir', default= '/home/chihchieh/retrival_anomaly_detection/mls_file/cq/shift', type=str, help='json_dir')
    parser.add_argument('--save-path', default='/home/chihchieh/retrival_anomaly_detection/mls_file/cq/shift_register', type=str, help='save_img_dir')
    parser.add_argument('--batch-size', default=300, type=int, help='batch size')
    
    return parser

def load_timg(file_name):
    """Loads the image with OpenCV and converts to torch.Tensor."""
    assert os.path.isfile(file_name), f"Invalid file {file_name}"  # nosec
    # load image with OpenCV
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    # convert image to torch tensor
    tensor = K.image_to_tensor(img, None).float() / 255.0
    return K.color.bgr_to_rgb(tensor)

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, y_true, y_pred,reduction= None):
        self.y_true = y_true
        self.y_pred = y_pred
    def forward(self):

        I = self.y_true
        J = self.y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('args script', parents=[get_args_parser()])
    args = parser.parse_args()
    s_img = args.source_img
    t_dir = args.save_path
    orig_dir = args.orig_dir
    m_file = args.mls_txt
    #os.makedirs(t_dir, exist_ok = True)
    
    #file = open(m_file,'r')
    #lines = [line.rstrip() for line in file]
    
    
    device = torch.device("cuda")
    ncc_loss = NCC
    registrator = KG.ImageRegistrator("similarity", loss_fn=F.l1_loss, lr=2e-1, pyramid_levels=4, num_iterations=2000).to(device)
    print(device)


    img2 = K.geometry.transform.resize(load_timg(s_img), (512, 512)).to(device)
    #for line in lines:
    homo_dict ={}
    count =0
    for dirPath, dirNames, fileNames in os.walk(orig_dir):
        print(dirPath)
        for f in fileNames:
            print(f)
            line = os.path.join(dirPath, f)
            img1 = K.geometry.transform.resize(load_timg(line), (512, 512)).to(device)
        
        
        
                
                    
            homo = registrator.register(img1, img2, output_intermediate_models=False)
            homo_dict[line] = homo.cpu().detach().numpy()
            #breakpoint()        
            with torch.no_grad():
            
                #print(m)
                timg_dst = KG.homography_warp(img1, homo, img2.shape[-2:])
                #print(m)
                final_img = K.tensor_to_image((timg_dst * 255.0).byte())
                final_path = line.replace(orig_dir,t_dir)
                assert final_path != line 
                final_dir = ('/').join(final_path.split('/')[:-1])
                os.makedirs(final_dir,exist_ok=True)
                cv2.imwrite(final_path,final_img) 
                   
            
    with open(m_file, 'wb') as final_file:
        pickle.dump(homo_dict, final_file)



        
                

            










    

