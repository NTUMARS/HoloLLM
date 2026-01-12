## HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning

[[Project Page](https://chuhaozhou99.github.io/HoloLLM)] [[Paper](https://arxiv.org/abs/2505.17645)] [[Model](https://drive.google.com/drive/folders/1ZUURrhBNvAswld4YRDfrAZ-mMZVjpV2N?usp=drive_link)] [[Data](#data)]

## News

- **2025.09.19** HoloLLM is accepted by **NeurIPS 2025**!🎉

## Contents

- [Install](#install)
- [Models](#models)
- [Data](#data)
- [Evaluation](#evaluation)
- [Training](#training)

### Install

1. Clone the repo into a local folder.

```bash
git clone https://github.com/NTUMARS/HoloLLM.git

cd HoloLLM
```

2. Install packages.

```bash
conda create -n holollm python=3.9 -y
conda activate holollm

pip install -r requirements.txt
```

3. Setup flash-attention-2

```bash
# Install flash-attn-2

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.2/flash_attn-2.6.2+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

pip install flash_attn-2.6.2+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```

4. Install pycocoevalcap for evaluation
```bash
cd ./eval/
git clone https://github.com/salaniz/pycocoevalcap.git
cd ./pycocoevalcap
pip install -e .

# Install java dependencies
sudo apt install openjdk-11-jre-headless
```
### Models

1. Universal Modality-Injection Projector (UMIP) of HoloLLM.

    Download the checkpoints of [UMIP]((https://drive.google.com/drive/folders/1yW_o8FgikuaKMHFrycVBM2BBQxoXNlxW?usp=drive_link)) and store them into *./checkpoints*.

2. Modality-Specific Tailored Encoders.

    Download the checkpoints of [modality-specific tailored encoders](https://drive.google.com/file/d/1ogH9stkAirfTS912Me5n-8MM9zjjphKC/view?usp=drive_link) and store them into *./modality_specific_encoders*.

3. LLaMA2-7B

    Download the checkpoints of [LLaMA2-7B](https://huggingface.co/meta-llama/Llama-2-7b/tree/main) and store the "consolidated.00.pth" into *./LLM_ckpt/llama2-7B*.

### Data

1. MMFi Dataset

    Download the [MMFi dataset](https://drive.google.com/drive/folders/1mzcM2AU7gZl5BAZikEjvdL-G0lm2EGjd) and the corresponding [RGB images](https://mmfi-dataset.oss-ap-southeast-1.aliyuncs.com/anonymized_rgb_images/all_images.zip).

    Organize the MMFi dataset into
    ```
    ./datasets/MMFi
    |-- E01
    |   |-- S01
    |   |   |-- A01
    |   |   |   |-- rgb_img (.png)
    |   |   |   |-- rgb
    |   |   |   |-- depth
    |   |   |   |-- mmwave
    |   |   |   |-- lidar   
    |   |   |   |-- wifi-csi
    |   |   |   |-- ...
    |   |   |-- A02
    |   |   |-- ...
    |   |   |-- A27
    |   |-- S02
    |   |-- ...
    |   |-- S10
    |-- E02
    |......
    |-- E03
    |......
    |-- E04
    |......
    ```
    
2. XRF55 Dataset

    Download the [XRF55 dataset](https://aiotgroup.github.io/XRF55/#). Please note that you need to request permission from the authors of XRF55 to access the Kinect videos (RGB modality).

    Organize the XRF55 dataset into
    ```
    ./datasets/XRF55
    |-- Scene1
    |   |-- Color
    |   |   |-- 03
    |   |   |   |-- 03_01_01
    |   |   |   |-- 03_01_02
    |   |   |   |-- 03_01_03
    |   |   |   |-- ... (PID_ActionID_SampleID)
    |   |   |-- 04
    |   |   |-- ...
    |   |   |-- 31
    |   |-- Depth
    |   |   |......
    |   |-- IR
    |   |   |......
    |   |-- WiFi
    |   |   |......
    |   |-- RFID
    |   |   |......
    |-- Scene2
    |   |......
    |-- Scene3
    |   |......
    |-- Scene4
    |   |......
    ```    

3. Textual Annotations of MMFi and XRF55

    Download the [textual annotations](https://drive.google.com/file/d/1WAFVeUqpIszMq5GU_MHb6sMyJQwvtv3w/view?usp=drive_link) of MMFi and XRF55 dataset and store them into *./datasets/textual_annotations/mmfi/* and *./datasets/textual_annotations/xrf55/*.
### Evaluation

1. Evaluate HoloLLM on MMFi
    
    Take "Random" setting as an example. Change the settings in *./eval/holollm_eval_mmfi.py*
    ```python
    # line 400

    # cross_env, cross_sub, random
    exp_settings = "random"
    pretrained_path = "./checkpoints/holollm_mmfi_random.pth"
    llm_type = "holollm_random_mmfi"
    base_path = "model./eval/holollm_mmfi_random/"
    llama_ckpt_dir = "./LLM_ckpt/llama2-7B"
    modality_list = ["mmfi_video", "mmfi_depth", "mmfi_mmwave", "mmfi_lidar", "mmfi_wifi"]
    ```
    Then, run the following command in the terminal:
    ```bash
    python ./eval/holollm_eval_mmfi.py
    ```
    

2. Evaluate HoloLLM on XRF55
    Take "Random" setting as an example. Change the settings in *./eval/holollm_eval_xrf55.py*
    ```python
    # line 285

    # cross_env, cross_sub, random
    exp_settings = "random"
    pretrained_path = "./checkpoints/holollm_xrf55_random.pth"
    llm_type = "holollm_random_xrf55"
    base_path = "./eval/holollm_xrf55_random/"
    llama_ckpt_dir = "./LLM_ckpt/llama2-7B"
    modality_list = ["xrf55_infra", "xrf55_wifi", "xrf55_rfid", "xrf55_depth", "xrf55_video"]
    ```
    Then, run the following command in the terminal:
    ```bash
    python ./eval/holollm_eval_xrf55.py
    ```
### Training

1. Training HoloLLM on MMFi

    Take "Random" setting as an example, directly run the following command in the terminal.
    ```bash
    bash ./scripts_sh/holollm_mmfi_random.sh
    ```

2. Training HoloLLM on XRF55

    Take "Random" setting as an example, directly run the following command in the terminal.
    ```bash
    bash ./scripts_sh/holollm_xrf55_random.sh
    ```


## Citation

```
@article{zhou2025holollm,
  title={HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning},
  author={Zhou, Chuhao and Yang, Jianfei},
  journal={arXiv preprint arXiv:2505.17645},
  year={2025}
}
```

## Acknowledgement

[Llama2](https://huggingface.co/meta-llama/Llama-2-7b), [OneLLM](https://github.com/csuhan/OneLLM), [Tokenpacker](https://github.com/CircleRadon/TokenPacker), [Honeybee](https://github.com/khanrc/honeybee), [ImageBind](https://github.com/facebookresearch/ImageBind), [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind), [MM-Fi](https://github.com/ybhbingo/MMFi_dataset), [XRF55](https://aiotgroup.github.io/XRF55/).


## License
This project is developed based on OneLLM and Llama 2, please refer to the [LLAMA 2 Community License](LICENSE_llama2).

## ⚠️ Trouble Shootings
1. "FileNotFoundError: [Errno 2] No such file or directory: 'java'" when run evaluation.

    ```bash
    sudo apt install openjdk-11-jre-headless
    ```
