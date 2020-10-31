# DeepSDM: Boundary-aware Pneumothorax Segmentation in Chest X-Ray Images
## Background
Pneumothorax can be caused by a blunt chest injury, damage from underlying lung disease, or most horrifying—it may occur for no obvious reason at all. On some occasions, a collapsed lung can be a life-threatening event.

Pneumothorax is usually diagnosed by a radiologist on a chest x-ray, and can sometimes be very difficult to confirm. An accurate AI algorithm to detect pneumothorax would be useful in a lot of clinical scenarios. AI could be used to triage chest radiographs for priority interpretation or to provide a more confident diagnosis for non-radiologists.

**In this repository, we provide a well-trained pneumothorax diagnosis model, which can perform accurate pneumothorax segmentation in chest X-ray images.**
![Introduction.png](https://github.com/wangyunpengbio/DeepSDM/raw/master/imgs/1-intro.png
)
## Requirement
Related packages:
```
Python >= 3.6
torch == 1.1.0
cuda >= 9.0
pydicom
opencv-python
pickle
```
The complete list of packages can be found in `requirements.txt`. We strongly recommend using `anaconda` for environment configuration.
```
conda create -n deepsdm python=3.6
conda activate deepsdm
pip install -r requirements.txt
```
## Code Structure

- imgs: Directory of the pictures used in README.md
- input: Directory of the demo `Dicom` files, the result images will also be stored in this directory
- code
  - 1-inference-save-pkl.py: the script for model inference and the results are saved in `pkl` format.
  - 2-process-pkl-to-png.py: the script used to generate results in `png` and `nii` format.
  - MaskBinarizers.py: codes needed in the binarization process
  - dataset
    - dataset.py: codes for PyTorch to read the dataset
  - experiments/demo
    - inference.yaml: configuration file for the demo inference
    - process.yaml: configuration file for the demo process
    - **checkpoint**: Directory of the weight of the well-trained DeepSDM.
    - **log**: Directory of the training log of DeepSDM.
      - **Because of the big size of the checkpoint and log file, these files can be downloaded from Google drive or Baidu drive. If you have trouble downloading big files, the minimum required files for this tutorial are `checkpoint/baseline_fold0.pth, checkpoint/baseline_fold1.pth, checkpoint/baseline_fold2.pth, checkpoint/baseline_fold3.pth, checkpoint/baseline_fold4.pth`**
      - Google drive：
      `https://drive.google.com/drive/folders/1HOtJEwZCCH7eXu-Bu3h_qTWKMTuPzedj?usp=sharing`
      `https://drive.google.com/drive/folders/1hxCvyfmWBsqIF_nswmFnFdClLNkgKvho?usp=sharing`
      - Baidu drive：`https://pan.baidu.com/s/1IYf_7BPkpCjjetcc2OFzLQ` password：`s3h2`


  - models: codes used to build the model
  - utils: Directory of some useful scripts
    - dicom_to_dataframe.py: extract the meta data from Dicom files
    - helpers.py: some useful functions
    - make_folds.py: split the dataset into folds
    - mask_functions.py: functions used to extract the mask from the annotation file
    - misc.py: some useful functions
    - prepare_png.py: convert the `Dicom` files to `png` format and prepare the dataset

## Usage

### Prerequisite
Before using the code in this repository, you should install the required packages and then download the weights of the well-trained DeepSDM from the cloud storage. **The weights should be put into the directory `experiments/demo/checkpoint` as mentioned above in the 'Code Structure' section if you follow this tutorial.**

### Step 1: Dataset preparation
Firstly, we need to convert the `Dicom` files to `png` format and prepare the dataset directory. For the sample data we uploaded in this repository, you can easily use the following command to prepare the dataset.

```
cd code/utils
python prepare_png.py -test_path ../../input/test-dcm
```
If you want to know all the parameters of the script, you can use the following command.
```
$ python prepare_png.py -h
usage: prepare_png.py [-h] [-train_path [TRAIN_PATH]] [-test_path [TEST_PATH]]
                      [-rle_path [RLE_PATH]] [-out_path [OUT_PATH]]
                      [-n_train [N_TRAIN]] [-img_size [IMG_SIZE]]
                      [-n_threads [N_THREADS]]

Prepare png dataset for pneumatorax

optional arguments:
  -h, --help            show this help message and exit
  -train_path [TRAIN_PATH]
                        directory with train
  -test_path [TEST_PATH]
                        directory with test
  -rle_path [RLE_PATH]  path for rle csv file
  -out_path [OUT_PATH]  path for saving dataset
  -n_train [N_TRAIN]    size of train dataset
  -img_size [IMG_SIZE]  image size
  -n_threads [N_THREADS]
                        number of using threads

```

### Step 2: Model inference
Secondly, we need to feed the images into the DeepSDM model for inference, and the inference result will be saved in the `pkl` format. The parameters of the model inference are specified by `-config_file` using the `yaml` file. You can easily perform model inference using the following commands. 

```
python 1-inference-save-pkl.py -config_file experiments/demo/inference.yaml
```
The details of the script can be found by the following command.
```
$ python 1-inference-save-pkl.py -h
usage: 1-inference-save-pkl.py [-h] [-config_file [CONFIG_FILE]]

Inference each model and save the output in the pkl file

optional arguments:
  -h, --help            show this help message and exit
  -config_file [CONFIG_FILE]
                        inference config file path
```

### Step 3: Result process
Finally, we need to process the `pkl` file and generate the results in `png` and `nii` format. The parameters of this process are specified by `-config_file` using the `yaml` file. You can easily process the `pkl` file using the following commands.
```
python 2-process-pkl-to-png.py -config_file experiments/demo/process.yaml
```
The details of the script can be found by the following command.
```
$ python 2-process-pkl-to-png.py -h
usage: 2-process-pkl-to-png.py [-h] [-config_file [CONFIG_FILE]]

Pneumothorax pipeline

optional arguments:
  -h, --help            show this help message and exit
  -config_file [CONFIG_FILE]
                        process config file path
```
If you follow every step in this tutorial, you will find the results in `input/dataset/postprocess_result`. The `png` files can be viewed directly and two demo chest x-ray images are shown in the figure below. The `*.nii.gz` can be viewed using the software [`itk-SNAP`](http://www.itksnap.org/pmwiki/pmwiki.php). 

![Demo-result.png](https://github.com/wangyunpengbio/DeepSDM/raw/master/imgs/2-demo.png
)

### Additional Tips: Log file check
If you want to further check the change of the related loss during training, we also provide log files and commands for you. **The log files should be downloaded and put into the directory `experiments/demo/log` as mentioned above in the 'Code Structure' section if you follow this tutorial.** Then you can use the following command to view the log files in `tensorboard`.
```
cd code/experiments/demo/log
tensorboard --logdir=.
```
Then, the `tensorboard` will be running on the port `6006` as default. You can view it by opening the browser and entering the URL `http://localhost:6006/`.

![tensorboard-log.png](https://github.com/wangyunpengbio/DeepSDM/raw/master/imgs/3-log.gif
)