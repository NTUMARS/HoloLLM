import sys
sys.path.append('./')
import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from fairscale.nn.model_parallel import initialize as fs_init
from util.misc import default_tensor_type
from util.misc import setup_for_distributed
import torchvision.transforms as transforms
from model.meta import MetaModel
from model.meta_holo import MetaHoloModel
from data.conversation_lib import conv_templates
from data import video_utils
import torchvision.transforms as transforms
from PIL import Image
import random
import warnings
from model.tokenizer import Tokenizer
from data import conversation_lib
import scipy.io as scio
import cv2

import argparse

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json

class MMFiEvalDataset(Dataset):
    def __init__(self, modality, cap_data_path, vqa_data_path) -> None:
        super().__init__()

        vqa_data_base_path = vqa_data_path
        self.cap_datas = json.load(open(cap_data_path))

        if modality == "mmfi_video":
            self.vqa_datas = json.load(open(vqa_data_base_path+'mmfivqa_rgb.json'))
        elif modality == "mmfi_depth":
            self.vqa_datas = json.load(open(vqa_data_base_path+'mmfivqa_depth.json'))
        elif modality == "mmfi_infra":
            self.vqa_datas = json.load(open(vqa_data_base_path+'mmfivqa_infra.json'))
        elif modality == "mmfi_mmwave":
            self.vqa_datas = json.load(open(vqa_data_base_path+'mmfivqa_mmwave.json'))
        elif modality == "mmfi_lidar":
            self.vqa_datas = json.load(open(vqa_data_base_path+'mmfivqa_lidar.json'))
        elif modality == "mmfi_wifi":
            self.vqa_datas = json.load(open(vqa_data_base_path+'mmfivqa_wifi.json'))
        else:
            raise NotImplementedError("Not Support for This Modality!")
        
        self.modality = modality

        # sort by video_id
        self.cap_datas = sorted(self.cap_datas, key=lambda x: x['video_id'])
        self.vqa_datas = sorted(self.vqa_datas, key=lambda x: x['video_id'])

        self.action_list = [
            'stretching and relaxing',
            'horizontal chest expansion',
            'vertical chest expansion',
            'left twist',
            'right twist',
            'mark time',
            'left limb extension',
            'right limb extension',
            'lunge toward left-front',
            'lunge toward right-front', 
            'both limb extension',
            'squat',
            'raising left hand',
            'raising right hand',
            'lunge toward left side',
            'lunge toward right side',
            'waving left hand',
            'waving right hand',
            'picking up things',
            'throwing toward left side',
            'throwing toward right side',
            'kicking toward left side',
            'kicking toward right side',
            'left body extension',
            'right body extension',
            'jumping up',
            'bowing'
        ]

        # test just 10 data
        # self.cap_datas = self.cap_datas[0:10]
        # self.vqa_datas = self.vqa_datas[0:10]

    def __len__(self):
        return len(self.cap_datas)

    def load_mmfi_video(self, data):

        to_tensor_transform = transforms.ToTensor()
        video_frame_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # Normalize using ImageNet stats
            ]
        )

        video_path = data['video_path']
        video_start_index = data['start_index']
        video_end_index = data['end_index']
        frame_list = list(range(video_start_index, video_end_index + 1))

        if len(frame_list) < 5:
            sample_frames = frame_list
            # If the list length is less than 5, pad it with the last integer
            while len(sample_frames) < 5:
                sample_frames.append(sample_frames[-1])
        else:
            sample_frames = self.sample_from_parts(frame_list)

        # load frames
        all_video = []
        frame_based_path = os.path.join(video_path, "rgb_img")
        for frame_index in sample_frames:
            
            frame_path = frame_based_path + "/frame" + str(frame_index).zfill(3) + ".png"
            frame = Image.open(frame_path).convert('RGB')
            frame_tensor = to_tensor_transform(frame)
            all_video.append(frame_tensor)  # [3, 480, 640] -> [3, 1, 480, 640], 1 for temporal dim
        
        all_video = [video_frame_transform(frame).unsqueeze(dim=1) for frame in all_video]
        all_video = video_utils.SpatialCrop(224, num_crops=3)(all_video)
        all_video = torch.stack(all_video, dim=0)

        return all_video[:,:,0]   # [15, 3, 1, 224, 224] -> [15, 3, 224, 224], similar to load_video
        # return all_video
    
    def load_mmfi_depth(self, data):

        depth_transform = transforms.Compose(
            [
                transforms.Resize(224),
            ]
        )

        # sample "data", use to test load function.
        video_path = data['video_path']
        video_start_index = data['start_index']
        video_end_index = data['end_index']
        frame_list = list(range(video_start_index, video_end_index + 1))

        if len(frame_list) < 5:
            sample_frames = frame_list
            # If the list length is less than 5, pad it with the last integer
            while len(sample_frames) < 5:
                sample_frames.append(sample_frames[-1])
        else:
            sample_frames = self.sample_from_parts(frame_list)

        all_depth = []
        frame_based_path = os.path.join(video_path, "depth")
        for frame_index in sample_frames:
            frame_path = frame_based_path + "/frame" + str(frame_index).zfill(3) + ".png"
            frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)  # followed mmfi
            frame = frame * 0.001  # Convert unit to meter, followed mmfi
            frame_tensor = torch.from_numpy(np.copy(frame))
            # treat depth image to rgb image by repeat
            # all_depth.append(frame_tensor.unsqueeze(dim=0).repeat(3,1,1).unsqueeze(dim=1))  # [480, 640] -> [3, 1, 480, 640], 1 for temporal dim
            all_depth.append(frame_tensor.unsqueeze(dim=0).unsqueeze(dim=0))

        all_depth = [depth_transform(frame) for frame in all_depth]
        all_depth = video_utils.SpatialCrop(224, num_crops=3)(all_depth)
        all_depth = torch.stack(all_depth, dim=0)

        # [15, 1, 1, 224, 224] -> [15, 1, 224, 224], similar to load_video, depth only have one channel
        return all_depth[:,:,0]  
        # return all_depth 
    
    def load_mmfi_infra(self, data):
        video_path = data['video_path']
        video_start_index = data['start_index']
        video_end_index = data['end_index']
        frame_list = list(range(video_start_index, video_end_index + 1))

        if len(frame_list) < 5:
            sample_frames = frame_list
            # If the list length is less than 5, pad it with the last integer
            while len(sample_frames) < 5:
                sample_frames.append(sample_frames[-1])
        else:
            sample_frames = self.sample_from_parts(frame_list)

        all_infra = []
        frame_based_path = os.path.join(video_path, "infra1")
        for frame_index in sample_frames:
            frame_path = frame_based_path + "/frame" + str(frame_index).zfill(3) + ".npy"
            frame = np.load(frame_path)  # [17, 2] 17 keypoints from infrared image
            frame_tensor = torch.from_numpy(np.copy(frame))
            all_infra.append(frame_tensor)

        all_infra = torch.stack(all_infra, dim=0)
        # [5, 17, 2]  5 frames, each frame contain the keypoints from the infrared images.
        # what the difference between infra1 and infra2? They seems to be all (17,2) points.
        return all_infra
        # return all_infra.unsqueeze(dim=1)

    def load_mmfi_lidar(self, data):
        video_path = data['video_path']
        video_start_index = data['start_index']
        video_end_index = data['end_index']
        frame_list = list(range(video_start_index, video_end_index + 1))

        if len(frame_list) < 5:
            sample_frames = frame_list
            # If the list length is less than 5, pad it with the last integer
            while len(sample_frames) < 5:
                sample_frames.append(sample_frames[-1])
        else:
            sample_frames = self.sample_from_parts(frame_list)

        all_lidar = []
        frame_based_path = os.path.join(video_path, "lidar")
        for frame_index in sample_frames:
            frame_path = frame_based_path + "/frame" + str(frame_index).zfill(3) + ".bin"
            with open(frame_path, 'rb') as f:
                raw_data = f.read()
                data_tmp = np.frombuffer(raw_data, dtype=np.float64)
                data_tmp = data_tmp.reshape(-1, 3)
            frame_tensor = torch.from_numpy(np.copy(data_tmp))
            all_lidar.append(frame_tensor)

        max_point_num = 1536   

        all_lidar_padded = []
        for data_item in all_lidar:
            if data_item.shape[0] < max_point_num:
                padded_tensor = torch.zeros((max_point_num-data_item.shape[0], data_item.shape[1]))
                data_item = torch.cat((data_item, padded_tensor), dim=0)
            elif data_item.shape[0] > max_point_num:
                data_item = data_item[:max_point_num,:]
            all_lidar_padded.append(data_item)

        all_lidar_padded = torch.stack(all_lidar_padded, dim=0)
        return all_lidar_padded
        # return all_lidar_padded.unsqueeze(dim=1)
    
    def load_mmfi_mmwave(self, data):

        video_path = data['video_path']
        video_start_index = data['start_index']
        video_end_index = data['end_index']
        frame_list = list(range(video_start_index, video_end_index + 1))

        if len(frame_list) < 5:
            sample_frames = frame_list
            # If the list length is less than 5, pad it with the last integer
            while len(sample_frames) < 5:
                sample_frames.append(sample_frames[-1])
        else:
            sample_frames = self.sample_from_parts(frame_list)

        all_mmwave = []
        frame_based_path = os.path.join(video_path, "mmwave")
        for frame_index in sample_frames:
            frame_path = frame_based_path + "/frame" + str(frame_index).zfill(3) + ".bin"
            with open(frame_path, 'rb') as f:
                raw_data = f.read()
                data_tmp = np.frombuffer(raw_data, dtype=np.float64)
                data_tmp = data_tmp.copy().reshape(-1, 5)
            frame_tensor = torch.from_numpy(np.copy(data_tmp))
            all_mmwave.append(frame_tensor)

        max_point_num = 64
        all_mmwave_padded = []
        for data_item in all_mmwave:
            if data_item.shape[0] < max_point_num:
                padded_tensor = torch.zeros((max_point_num-data_item.shape[0], data_item.shape[1]))
                data_item = torch.cat((data_item, padded_tensor), dim=0)
            elif data_item.shape[0] > max_point_num:
                data_item = data_item[:max_point_num,:]
            all_mmwave_padded.append(data_item)
        all_mmwave_padded = torch.stack(all_mmwave_padded, dim=0)
        return all_mmwave_padded
        # return all_mmwave_padded.unsqueeze(dim=1)

    def load_mmfi_wifi(self, data):
        video_path = data['video_path']
        video_start_index = data['start_index']
        video_end_index = data['end_index']
        frame_list = list(range(video_start_index, video_end_index + 1))

        if len(frame_list) < 5:
            sample_frames = frame_list
            # If the list length is less than 5, pad it with the last integer
            while len(sample_frames) < 5:
                sample_frames.append(sample_frames[-1])
        else:
            sample_frames = self.sample_from_parts(frame_list)

        all_wifi = []
        frame_based_path = os.path.join(video_path, "wifi-csi")
        for frame_index in sample_frames:
            frame_path = frame_based_path + "/frame" + str(frame_index).zfill(3) + ".mat"

            data_mat = scio.loadmat(frame_path)['CSIamp']
            data_mat[np.isinf(data_mat)] = np.nan
            for i in range(10):  # 32
                temp_col = data_mat[:, :, i]
                nan_num = np.count_nonzero(temp_col != temp_col)
                if nan_num != 0:
                    temp_not_nan_col = temp_col[temp_col == temp_col]
                    temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()  
            data_mat = (data_mat - np.min(data_mat)) / (np.max(data_mat) - np.min(data_mat))
            data_frame = np.array(data_mat)
            frame_tensor = torch.from_numpy(np.copy(data_frame))

            all_wifi.append(frame_tensor)
        
        all_wifi = torch.stack(all_wifi, dim=0)
        # wifi for each frame has same size [3, 114, 10]
        # the output will be [5, 3, 114, 10]
        return all_wifi
        # return all_wifi.unsqueeze(dim=1)
    
    def __getitem__(self, index):
        cap_data = self.cap_datas[index]
        vqa_data = self.vqa_datas[index]

        if self.modality == "mmfi_video":
            all_video = self.load_mmfi_video(cap_data)
        elif self.modality == "mmfi_depth":
            all_video = self.load_mmfi_depth(cap_data)
        elif self.modality == "mmfi_infra":
            all_video = self.load_mmfi_infra(cap_data)
        elif self.modality == "mmfi_lidar":
            all_video = self.load_mmfi_lidar(cap_data)
        elif self.modality == "mmfi_mmwave":
            all_video = self.load_mmfi_mmwave(cap_data)
        elif self.modality == "mmfi_wifi":
            all_video = self.load_mmfi_wifi(cap_data)
        else:
            raise NotImplementedError("Not Support for This Modality!")

        # HAR classification
        action_prompt = "Human: What is the human's action in the video?"
        action_label = int(cap_data["video_path"].split("/")[-1][1:]) - 1  # label is [0-26]
        action_label = torch.tensor(action_label, dtype=torch.int64)

        # HAR VQA
        # vqa_question = vqa_data['conversations'][2]["value"]
        # vqa_answer = vqa_data['conversations'][3]["value"]
        
        # new dataset
        # vqa_question = vqa_data['conversations'][0]["value"]
        vqa_question = "What is the human's action according to the input data?\n You must answer the question by only selecting a word or phrase from the attached Action List without any other words."

        vqa_question = vqa_question + ("\Action List: [")
        for item in self.action_list:
            vqa_question = vqa_question + "'" + item + "', "
        vqa_question = vqa_question[:-2]  # remove the last ", "
        vqa_question = vqa_question + "]"

        vqa_answer = vqa_data['conversations'][1]["value"]
        video_id = vqa_data["video_id"]

        # Caption
        caption_prompt = "Human: What is the human's action in the video?"

        return all_video, action_prompt, action_label, vqa_question, vqa_answer, caption_prompt, video_id, self.modality

    def sample_from_parts(self, frame_list):
        """
            For example, a video with 7 frames
            divide into 1,1,1,1,3
            the first 4 frames are selected
            randomly selected one frame from the last part
        """
        # Determine the size of each part
        part_size = len(frame_list) // 5
        remainder = len(frame_list) % 5
        
        # Split the list into 5 parts
        parts = []
        start = 0
        for i in range(4):
            parts.append(frame_list[start:start + part_size])
            start += part_size
        
        # Assign the remaining elements to the last part
        parts.append(frame_list[start:])  # Last part includes the remainder

        # Sample one integer from each part
        sampled_integers = [random.choice(part) for part in parts if part]
        
        return sampled_integers   

    
if __name__ == "__main__":
    # cross_env, cross_sub, random
    exp_settings = "random"
    pretrained_path = "./checkpoints/holollm_mmfi_random.pth"
    llm_type = "holollm_random_mmfi"
    base_path = "./eval/holollm_mmfi_random/"
    llama_ckpt_dir = "./LLM_ckpt/llama2-7B"
    # modality_list = ["mmfi_video", "mmfi_depth", "mmfi_mmwave", "mmfi_lidar", "mmfi_wifi"]
    modality_list = ["mmfi_wifi"]
    port = "tcp://127.0.0.1:24682"

    if exp_settings == "random":
        vqa_data_path = "./datasets/textual_annotations/mmfi/mmfivqa/test_random_full/"
        cap_data_path = "./datasets/textual_annotations/mmfi/mmficap/mmfi_test_random_full.json"
        cap_gt_file = "./datasets/textual_annotations/cap_gts/mmfi_cap_random_full_gt.json"
    elif exp_settings == "cross_sub":
        vqa_data_path = "./datasets/textual_annotations/mmfi/mmfivqa/test_cs_full/"
        cap_data_path = "./datasets/textual_annotations/mmfi/mmficap/mmfi_test_cs_full.json"
        cap_gt_file = "./datasets/textual_annotations/cap_gts/mmfi_cap_cs_full_gt.json"
    elif exp_settings == "cross_env":
        vqa_data_path = "./datasets/textual_annotations/mmfi/mmfivqa/test_ce_full/"
        cap_data_path = "./datasets/textual_annotations/mmfi/mmficap/mmfi_test_ce_full.json"
        cap_gt_file = "./datasets/textual_annotations/cap_gts/mmfi_cap_ce_full_gt.json"
    

    mp.set_start_method("spawn")

    dist.init_process_group(
        backend="nccl", rank=0, world_size=1,
        init_method=port)
    fs_init.initialize_model_parallel(1)
    torch.cuda.set_device(0)
    torch.manual_seed(1)
    np.random.seed(1)
    # set the print behavior.
    setup_for_distributed(True)

    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16
    }['fp16']
    with default_tensor_type(dtype=target_dtype, device="cuda"):
       

        model = MetaHoloModel(llm_type, "config/llama2/7B.json", llama_ckpt_dir, "config/llama2/tokenizer.model")


    print("Loading pretrained weights ...")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)  # only load projector
    print("load result:\n", msg)

    model.half().cuda()
    model.eval()
    print(f"Model = {str(model)}")

    def multi_modal_generate(images, inps, modality):
        images = images.cuda().to(target_dtype)

        prompts = []
        for inp in inps:
            conv = conv_templates["v1"].copy()        
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())

        with torch.cuda.amp.autocast(dtype=target_dtype):
            # original temp = 0.1
            responses = model.generate(prompts, images, 128, temperature=0.1, top_p=0.75, modal=[modality])
            # responses = model.generate(prompts, images, 128, temperature=0.01, top_p=0.75, modal=[modality])
            outputs = []
            for response, prompt in zip(responses, prompts):
                response = response[len(prompt):].split('###')[0]
                response = response.strip()
                outputs.append(response)
        return outputs

    acc_base_path = base_path + "acc/"
    vqa_base_path = base_path + "vqa/"
    cap_base_path = base_path + "cap/"
    coco_cap_base_path = base_path + "cap_coco/"
    
    
    for cur_modality in modality_list:
        # Acc path
        if cur_modality == "mmfi_video":
            acc_answer_path = acc_base_path + "eval_mmfi_video.txt"
        elif cur_modality == "mmfi_depth":
            acc_answer_path = acc_base_path + "eval_mmfi_depth.txt"
        elif cur_modality == "mmfi_infra":
            acc_answer_path = acc_base_path + "eval_mmfi_infra.txt"
        elif cur_modality == "mmfi_lidar":
            acc_answer_path = acc_base_path + "eval_mmfi_lidar.txt"
        elif cur_modality == "mmfi_mmwave":
            acc_answer_path = acc_base_path + "eval_mmfi_mmwave.txt"
        elif cur_modality == "mmfi_wifi":
            acc_answer_path = acc_base_path + "eval_mmfi_wifi.txt"
        else:
            sys.exit(-1)

        # VQA path
        if cur_modality == "mmfi_video":
            vqa_answer_path = vqa_base_path + "eval_mmfi_video.json"
        elif cur_modality == "mmfi_depth":
            vqa_answer_path = vqa_base_path + "eval_mmfi_depth.json"
        elif cur_modality == "mmfi_infra":
            vqa_answer_path = vqa_base_path + "eval_mmfi_infra.json"
        elif cur_modality == "mmfi_lidar":
            vqa_answer_path = vqa_base_path + "eval_mmfi_lidar.json"
        elif cur_modality == "mmfi_mmwave":
            vqa_answer_path = vqa_base_path + "eval_mmfi_mmwave.json"
        elif cur_modality == "mmfi_wifi":
            vqa_answer_path = vqa_base_path + "eval_mmfi_wifi.json"
        else:
            sys.exit(-1)

        # Cap path
        if cur_modality == "mmfi_video":
            cap_answer_path = cap_base_path + "eval_mmfi_video.json"
        elif cur_modality == "mmfi_depth":
            cap_answer_path = cap_base_path + "eval_mmfi_depth.json"
        elif cur_modality == "mmfi_infra":
            cap_answer_path = cap_base_path + "eval_mmfi_infra.json"
        elif cur_modality == "mmfi_lidar":
            cap_answer_path = cap_base_path + "eval_mmfi_lidar.json"
        elif cur_modality == "mmfi_mmwave":
            cap_answer_path = cap_base_path + "eval_mmfi_mmwave.json"
        elif cur_modality == "mmfi_wifi":
            cap_answer_path = cap_base_path + "eval_mmfi_wifi.json"
        else:
            sys.exit(-1)
    
        # cap score path
        if cur_modality == "mmfi_video":
            cap_score_path = cap_base_path + "video_cap_score.txt"
        elif cur_modality == "mmfi_depth":
            cap_score_path = cap_base_path + "depth_cap_score.txt"
        elif cur_modality == "mmfi_infra":
            cap_score_path = cap_base_path + "infra_cap_score.txt"
        elif cur_modality == "mmfi_lidar":
            cap_score_path = cap_base_path + "lidar_cap_score.txt"
        elif cur_modality == "mmfi_mmwave":
            cap_score_path = cap_base_path + "mmwave_cap_score.txt"
        elif cur_modality == "mmfi_wifi":
            cap_score_path = cap_base_path + "wifi_cap_score.txt"
        else:
            sys.exit(-1)

        # COCO Cap path
        if cur_modality == "mmfi_video":
            coco_cap_answer_path = coco_cap_base_path + "eval_mmfi_video.json"
        elif cur_modality == "mmfi_depth":
            coco_cap_answer_path = coco_cap_base_path + "eval_mmfi_depth.json"
        elif cur_modality == "mmfi_infra":
            coco_cap_answer_path = coco_cap_base_path + "eval_mmfi_infra.json"
        elif cur_modality == "mmfi_lidar":
            coco_cap_answer_path = coco_cap_base_path + "eval_mmfi_lidar.json"
        elif cur_modality == "mmfi_mmwave":
            coco_cap_answer_path = coco_cap_base_path + "eval_mmfi_mmwave.json"
        elif cur_modality == "mmfi_wifi":
            coco_cap_answer_path = coco_cap_base_path + "eval_mmfi_wifi.json"
        else:
            sys.exit(-1)

        os.makedirs(os.path.dirname(acc_answer_path), exist_ok=True)
        os.makedirs(os.path.dirname(vqa_answer_path), exist_ok=True)
        os.makedirs(os.path.dirname(cap_answer_path), exist_ok=True)
        os.makedirs(os.path.dirname(coco_cap_answer_path), exist_ok=True)
    
        print("Starting...")
        dataset = MMFiEvalDataset(modality = cur_modality, vqa_data_path=vqa_data_path, cap_data_path=cap_data_path)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)

        act_correct = 0
        vqa_correct = 0
        vqa_predictions = []
        cap_predictions = []


        with torch.no_grad():
            for data in tqdm(dataloader):
                images, act_prompts, act_labels, vqa_questions, vqa_answers, cap_prompts, image_ids, modality = data
                # modality = modality[0]
                modality = modality[0].split("_")[-1]
                # Action
                images_cuda = images.cuda()
                act_preds = model.predict_action(act_prompts, images_cuda, modality)
                pred_classes = torch.argmax(act_preds, dim=1)
                correct_predictions = (pred_classes.detach().cpu() == act_labels)
                act_correct += correct_predictions.sum().item()

                # VQA
                vqa_preds = multi_modal_generate(images, vqa_questions, modality)
                for question, pred, question_id, answer in zip(vqa_questions, vqa_preds, image_ids, vqa_answers):
                    vqa_predictions.append({'question_id': question_id, 'answer': pred, 'gt_answer': answer})
                    pred = pred.strip().lower()
                    answer = answer.strip().lower()
                    print(pred)
                    if pred != "":
                        if (pred in answer) or (answer in pred):
                            vqa_correct += 1
                
                # # Caption
                cap_preds = multi_modal_generate(images, cap_prompts, modality)
                for question, pred, image_id in zip(cap_prompts, cap_preds, image_ids):
                    cap_predictions.append({'image_id': image_id, 'caption': pred})
        
        # action
        act_acc = float(act_correct) / len(dataset)
        vqa_acc = float(vqa_correct) / len(dataset)
        
        act_line = "The action accuracy: " + str(act_acc)
        print(act_line)
        vqa_line = "The vqa accuracy: " + str(vqa_acc)
        print(vqa_line)

        with open(acc_answer_path, "w") as file:
            file.write(act_line + "\n")
            file.write(vqa_line + "\n")
        
        with open(vqa_answer_path, 'w') as file:
            json.dump(vqa_predictions, file)

        with open(cap_answer_path, 'w') as file:
            json.dump(cap_predictions, file)

        # cap to coco
        predictions = []
        data = json.load(open(cap_answer_path))
        for idx, data_item in enumerate(data):
            print(idx)
            image_id = int(data_item['image_id'])
            ans_cap = data_item['caption']

            predictions.append({'image_id': image_id, 'caption': ans_cap})
        # new_data = {}
        # new_data["annotations"] = data
        with open(coco_cap_answer_path, 'w') as f:
            json.dump(predictions, f)
        
        # eval cap metric

        annotation_file = cap_gt_file
        results_file = coco_cap_answer_path
        coco = COCO(annotation_file)
        coco_result = coco.loadRes(results_file)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate()


        cap_score_file = open(cap_score_path, "w")
        for metric, score in coco_eval.eval.items():
            print(f'{metric}: {score:.3f}')
            print(f'{metric}: {score:.3f}', file=cap_score_file)