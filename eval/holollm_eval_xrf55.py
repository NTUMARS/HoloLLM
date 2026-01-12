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

class XRF55EvalDataset(Dataset):
    def __init__(self, modality, cap_data_path, vqa_data_path) -> None:
        super().__init__()

        vqa_data_base_path = vqa_data_path
        self.cap_datas = json.load(open(cap_data_path))

        if modality == "xrf55_video":
            self.vqa_datas = json.load(open(vqa_data_base_path+'xrfvqa_xrf55_video.json'))
        elif modality == "xrf55_depth":
            self.vqa_datas = json.load(open(vqa_data_base_path+'xrfvqa_xrf55_depth.json'))
        elif modality == "xrf55_infra":
            self.vqa_datas = json.load(open(vqa_data_base_path+'xrfvqa_xrf55_infra.json'))
        elif modality == "xrf55_wifi":
            self.vqa_datas = json.load(open(vqa_data_base_path+'xrfvqa_xrf55_wifi.json'))
        elif modality == "xrf55_rfid":
            self.vqa_datas = json.load(open(vqa_data_base_path+'xrfvqa_xrf55_rfid.json'))
        else:
            raise NotImplementedError("Not Support for This Modality!")
        
        self.modality = modality

        # sort by video_id
        self.cap_datas = sorted(self.cap_datas, key=lambda x: x['video_id'])
        self.vqa_datas = sorted(self.vqa_datas, key=lambda x: x['video_id'])

        self.action_list = [
            'carrying weight',
            'mopping the floor',
            'cutting',
            'wearing hat',
            'using a phone',
            'throw something',
            'put something on the table',
            'put on clothing',
            'picking',
            'drinking', 
            'smoking',
            'eating',
            'brushing teeth',
            'blow dry hair',
            'brush hair',
            'shake hands',
            'hugging',
            'hand something to someone',
            'kick someone',
            'hit someone with something',
            'choke someone’s neck',
            'push someone',
            'body weight squats',
            'tai chi',
            'boxing',
            'weightlifting',
            'hula hooping',
            'jump rope',
            'jumping jack',
            'high leg lift',
            'waving',
            'clap hands',
            'fall on the floor',
            'jumping',
            'running',
            'sitting down',
            'standing up',
            'turning',
            'walking',
            'stretch oneself',
            'pat on shoulder',
            'playing erhu',
            'playing ukulele',
            'playing drum',
            'stomping',
            'shaking head',
            'nodding',
            'draw circles',
            'draw a cross',
            'pushing',
            'pulling',
            'swipe left',
            'swipe right',
            'swipe up',
            'swipe down'
        ]

        # test just 10 data
        # self.cap_datas = self.cap_datas[0:10]
        # self.vqa_datas = self.vqa_datas[0:10]

    def __len__(self):
        return len(self.cap_datas)

    def load_xrf55_video(self, data):


        to_tensor_transform = transforms.ToTensor()
        video_frame_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # Normalize using ImageNet stats
            ]
        )
        video_path = data['video_path']
        frame_list = os.listdir(video_path)

        # load frames
        all_video = []
        frame_based_path = video_path
        for frame_index in frame_list:
            frame_path = frame_based_path + "/" + frame_index
            frame = Image.open(frame_path).convert('RGB')
            frame_tensor = to_tensor_transform(frame)
            # all_video.append(frame_tensor.unsqueeze(dim=1))  # [3, 480, 640] -> [3, 1, 480, 640], 1 for temporal dim
            all_video.append(frame_tensor)  # [3, 480, 640] -> [3, 1, 480, 640], 1 for temporal dim
        # if the length < 10, use the last one to append to 10
        all_video = [video_frame_transform(frame).unsqueeze(dim=1) for frame in all_video]
        while len(all_video) < 10:
            all_video.append(all_video[-1])
        all_video = video_utils.SpatialCrop(224, num_crops=3)(all_video)
        all_video = torch.stack(all_video, dim=0)

        return all_video[:,:,0]   # [15, 3, 1, 224, 224] -> [30, 3, 224, 224]
    
    def load_xrf55_depth(self, data):

        depth_transform = transforms.Compose(
            [
                transforms.Resize(224),
            ]
        )

        # sample "data", use to test load function.
        video_path = data['video_path']
        video_path = video_path.replace("Color", "Depth")
        frame_list = os.listdir(video_path)

        all_depth = []
        frame_based_path = video_path
        for frame_index in frame_list:
            frame_path = frame_based_path + "/" + frame_index
            frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)  # followed mmfi
            frame = frame * 0.001  # Convert unit to meter, followed mmfi
            frame_tensor = torch.from_numpy(np.copy(frame))
            all_depth.append(frame_tensor.unsqueeze(dim=0).unsqueeze(dim=0))

        all_depth = [depth_transform(frame) for frame in all_depth]
        while len(all_depth) < 10:
            all_depth.append(all_depth[-1])
        all_depth = video_utils.SpatialCrop(224, num_crops=3)(all_depth)
        all_depth = torch.stack(all_depth, dim=0)

        # [30, 1, 1, 224, 224] -> [30, 1, 224, 224], similar to load_video, depth only have one channel
        return all_depth[:,:,0] 
    
    def load_xrf55_infra(self, data):
        to_tensor_transform = transforms.ToTensor()
        infra_transform = transforms.Compose(
            [
                transforms.Resize(224),
            ]
        )

        video_path = data['video_path']
        video_path = video_path.replace("Color", "IR")
        frame_list = os.listdir(video_path)

        # load frames
        all_infra = []
        frame_based_path = video_path
        for frame_index in frame_list:
            frame_path = frame_based_path + "/" + frame_index
            # frame = Image.open(frame_path).convert('RGB')
            frame = Image.open(frame_path)
            frame_tensor = to_tensor_transform(frame)
            all_infra.append(frame_tensor.unsqueeze(dim=1))  # [1, 512, 512] -> [1, 1, 512, 512], 1 for temporal dim
        
        all_infra = [infra_transform(frame) for frame in all_infra]
        while len(all_infra) < 10:
            all_infra.append(all_infra[-1])
        all_infra = video_utils.SpatialCrop(224, num_crops=3)(all_infra)
        all_infra = torch.stack(all_infra, dim=0)

        return all_infra[:,:,0]   # [30, 1, 1, 224, 224] -> [30, 1, 224, 224]

    def load_xrf55_wifi(self, data):
        video_path = data['video_path']
        if ".npy" in video_path:
            video_path = video_path.replace("Color", "WiFi")[0:-1]
        else: # cap
            video_path = video_path.replace("Color", "WiFi")[0:-1] + ".npy"

        all_wifi = np.load(video_path)
        frame_tensor = torch.from_numpy(np.copy(all_wifi))
        frame_tensor = frame_tensor.view(9, 30, 5, 200).permute(2, 0, 1, 3)
        all_wifi = frame_tensor  # [5, 9, 30, 200]
        return all_wifi
    
    def load_xrf55_rfid(self, data):
        video_path = data['video_path']
        if ".npy" in video_path:
            video_path = video_path.replace("Color", "RFID")[0:-1]
        else: # cap
            video_path = video_path.replace("Color", "RFID")[0:-1] + ".npy"

        all_rfid = np.load(video_path)
        frame_tensor = torch.from_numpy(np.copy(all_rfid))  # [23, 148] 23 tags, each contain 148 features.
        all_rfid = frame_tensor  # [23, 148]
        return all_rfid
    
    def __getitem__(self, index):
        cap_data = self.cap_datas[index]
        vqa_data = self.vqa_datas[index]

        if self.modality == "xrf55_video":
            all_video = self.load_xrf55_video(cap_data)
        elif self.modality == "xrf55_depth":
            all_video = self.load_xrf55_depth(cap_data)
        elif self.modality == "xrf55_infra":
            all_video = self.load_xrf55_infra(cap_data)
        elif self.modality == "xrf55_wifi":
            all_video = self.load_xrf55_wifi(cap_data)
        elif self.modality == "xrf55_rfid":
            all_video = self.load_xrf55_rfid(cap_data)
        else:
            raise NotImplementedError("Not Support for This Modality!")

        # HAR classification
        action_prompt = "Human: What is the human's action in the video?"
        action_label_tmp = cap_data["video_path"].split("/")[-2]
        action_label = int(action_label_tmp.split("_")[1]) - 1
        action_label = torch.tensor(action_label, dtype=torch.int64)

        # HAR VQA
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
    
if __name__ == "__main__":

    # cross_env, cross_sub, random
    exp_settings = "random"
    pretrained_path = "./checkpoints/holollm_xrf55_random.pth"
    llm_type = "holollm_random_xrf55"
    base_path = "./eval/holollm_xrf55_random/"
    llama_ckpt_dir = "./LLM_ckpt/llama2-7B"
    modality_list = ["xrf55_infra", "xrf55_wifi", "xrf55_rfid", "xrf55_depth", "xrf55_video"]
    port = "tcp://127.0.0.1:24582"
    
    if exp_settings == "random":
        vqa_data_path = "./datasets/textual_annotations/xrf55/xrf55vqa/test_random_full/"
        cap_data_path = "./datasets/textual_annotations/xrf55/xrf55cap/xrf55_test_random_full.json"
        cap_gt_file = "./datasets/textual_annotations/cap_gts/xrf55_cap_random_full_gt.json"
    elif exp_settings == "cross_sub":
        vqa_data_path = "./datasets/textual_annotations/xrf55/xrf55vqa/test_cs_full/"
        cap_data_path = "./datasets/textual_annotations/xrf55/xrf55cap/xrf55_test_cs_full.json"
        cap_gt_file = "./datasets/textual_annotations/cap_gts/xrf55_cap_cs_full_gt.json"
    elif exp_settings == "cross_env":
        vqa_data_path = "./datasets/textual_annotations/xrf55/xrf55vqa_v2/test_ce_full/"
        cap_data_path = "./datasets/textual_annotations/xrf55/xrf55cap/xrf55_test_ce_full.json"
        cap_gt_file = "./datasets/textual_annotations/cap_gts/xrf55_cap_ce_full_gt.json"
        
    
    

    mp.set_start_method("spawn")
    # dist.init_process_group(
    #     backend="nccl", rank=0, world_size=1,
    #     init_method=f"tcp://127.0.0.1:23960")
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
    msg = model.load_state_dict(checkpoint, strict=False)
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
            responses = model.generate(prompts, images, 128, temperature=0.7, top_p=0.95, modal=[modality])
           
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
        if cur_modality == "xrf55_video":
            acc_answer_path = acc_base_path + "eval_xrf55_video.txt"
        elif cur_modality == "xrf55_depth":
            acc_answer_path = acc_base_path + "eval_xrf55_depth.txt"
        elif cur_modality == "xrf55_infra":
            acc_answer_path = acc_base_path + "eval_xrf55_infra.txt"
        elif cur_modality == "xrf55_wifi":
            acc_answer_path = acc_base_path + "eval_xrf55_wifi.txt"
        elif cur_modality == "xrf55_rfid":
            acc_answer_path = acc_base_path + "eval_xrf55_rfid.txt"
        else:
            sys.exit(-1)

        # VQA path
        if cur_modality == "xrf55_video":
            vqa_answer_path = vqa_base_path + "eval_xrf55_video.json"
        elif cur_modality == "xrf55_depth":
            vqa_answer_path = vqa_base_path + "eval_xrf55_depth.json"
        elif cur_modality == "xrf55_infra":
            vqa_answer_path = vqa_base_path + "eval_xrf55_infra.json"
        elif cur_modality == "xrf55_wifi":
            vqa_answer_path = vqa_base_path + "eval_xrf55_wifi.json"
        elif cur_modality == "xrf55_rfid":
            vqa_answer_path = vqa_base_path + "eval_xrf55_rfid.json"
        else:
            sys.exit(-1)

        # Cap path
        if cur_modality == "xrf55_video":
            cap_answer_path = cap_base_path + "eval_xrf55_video.json"
        elif cur_modality == "xrf55_depth":
            cap_answer_path = cap_base_path + "eval_xrf55_depth.json"
        elif cur_modality == "xrf55_infra":
            cap_answer_path = cap_base_path + "eval_xrf55_infra.json"
        elif cur_modality == "xrf55_wifi":
            cap_answer_path = cap_base_path + "eval_xrf55_wifi.json"
        elif cur_modality == "xrf55_rfid":
            cap_answer_path = cap_base_path + "eval_xrf55_rfid.json"
        else:
            sys.exit(-1)

        # cap score path
        if cur_modality == "xrf55_video":
            cap_score_path = cap_base_path + "video_cap_score.txt"
        elif cur_modality == "xrf55_depth":
            cap_score_path = cap_base_path + "depth_cap_score.txt"
        elif cur_modality == "xrf55_infra":
            cap_score_path = cap_base_path + "infra_cap_score.txt"
        elif cur_modality == "xrf55_wifi":
            cap_score_path = cap_base_path + "wifi_cap_score.txt"
        elif cur_modality == "xrf55_rfid":
            cap_score_path = cap_base_path + "rfid_cap_score.txt"
        else:
            sys.exit(-1)


        # COCO Cap path
        if cur_modality == "xrf55_video":
            coco_cap_answer_path = coco_cap_base_path + "eval_xrf55_video.json"
        elif cur_modality == "xrf55_depth":
            coco_cap_answer_path = coco_cap_base_path + "eval_xrf55_depth.json"
        elif cur_modality == "xrf55_infra":
            coco_cap_answer_path = coco_cap_base_path + "eval_xrf55_infra.json"
        elif cur_modality == "xrf55_wifi":
            coco_cap_answer_path = coco_cap_base_path + "eval_xrf55_wifi.json"
        elif cur_modality == "xrf55_rfid":
            coco_cap_answer_path = coco_cap_base_path + "eval_xrf55_rfid.json"
        else:
            sys.exit(-1)

        os.makedirs(os.path.dirname(acc_answer_path), exist_ok=True)
        os.makedirs(os.path.dirname(vqa_answer_path), exist_ok=True)
        os.makedirs(os.path.dirname(cap_answer_path), exist_ok=True)
        os.makedirs(os.path.dirname(coco_cap_answer_path), exist_ok=True)
    
        print("Starting...")
        dataset = XRF55EvalDataset(modality = cur_modality, vqa_data_path=vqa_data_path, cap_data_path=cap_data_path)
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
                    # if (pred in answer) or (answer in pred):
                    #     vqa_correct += 1
                    if pred != "":
                        if (pred in answer) or (answer in pred):
                            vqa_correct += 1
                
                # Caption
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