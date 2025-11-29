# Sinkhorn-based-Quantization-for-Reversed-Disease-Progress
This is the official implementation of our work "Sinkhorn-based Quantization for Reversed Disease Progress: Further Investigations."


## Warning!!
Our flow matching model is trained on registered CT images. We prepared some registered CT images from the [CQ500 dataset](https://drive.google.com/file/d/1JG5Wk8Mm9f2ADQbSZOQdujt5OFk3FNtl/view?usp=sharing) released from [here](https://www.qure.ai/evidence/validation-of-deep-learning-algorithms-for-detection-of-critical-findings-in-head-ct-scans). To test the performances of our model on your own dataset. You need to do:

1. Set window width=80, window center = 40, transfer the Dicom files to PNG files. (Use the [dcm_to_png](https://github.com/chihchiehchen/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization/blob/main/utils/transfer_to_png.py) function.)
2. Rigid register your PNG files to anyone of the registered CT images provided the above.
   ```
   python register_mls.py --orig-dir=<the png img dir> --save-path=<final save dir> 
   ```
