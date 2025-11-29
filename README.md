# Sinkhorn-based-Quantization-for-Reversed-Disease-Progress
This is the official implementation of our work "Sinkhorn-based Quantization for Reversed Disease Progress: Further Investigations."


## Warning!!
Our flow matching model is trained on registered CT images. We prepared some registered CT images from the [CQ500 dataset](https://drive.google.com/file/d/1i0be9wi6uCoXlllZDRmD0qZc6VNKzfBN/view?usp=sharing) released from [here](https://www.qure.ai/evidence/validation-of-deep-learning-algorithms-for-detection-of-critical-findings-in-head-ct-scans). To test the performances of our model on your own dataset. You need to do:

1. Set window width=80, window center = 40, transfer the Dicom files to PNG files. (Use the [dcm_to_png](https://github.com/chihchiehchen/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization/blob/main/utils/transfer_to_png.py) function.)
2. Rigid register your PNG files to anyone of the registered CT images in the directory register_img.
   ```
   python register_mls.py --orig-dir=<the png img dir> --save-path=<final save dir> 
   ```
The checkpoints can be download from the [link](https://drive.google.com/file/d/1hRmcZ29J0VYAaBk8C5UtSkHD6pq6Cd00/view?usp=sharing), where checkpoint.pth.epoch413_curr is the checkpoint for OptVQ, and otcfmnet_CXR_weights_step_20000.pt is the checkpoint for the flow matching model.

To visualize the reversed disease progress, run:
'''
python visualize_inference.py   --config_file ./optvq_config.yaml --checkpoint_path ./checkpoint.pth.epoch413_curr --latent-path otcfmnet_CXR_weights_step_20000.pt  --png_dir <Your ing dirs > --output_dir <your output dir>
'''
