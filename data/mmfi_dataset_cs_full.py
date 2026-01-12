import warnings
from typing import Optional
import os
import copy
import numpy as np
import random
from torch.utils.data import Sampler, Dataset
import torch
from PIL import Image
import torchvision.transforms as transforms
import json
from model.tokenizer import Tokenizer
from . import conversation_lib

from . import video_utils
from .data_utils import T_random_resized_crop, transform_pairimg_train, make_audio_features

import scipy.io as scio
import cv2

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"


class ConversationGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.header = f"{conversation_lib.default_conversation.system}\n\n"
        self._probe_tokenizer_style()

    def _probe_tokenizer_style(self):
        """
        Given a sentence, e.g. "My darling", some tokenizers will make the space a seperate token,
        while some others will merge the space into the next word, forming a token representing " darling".
        Knowing which style the tokenizer takes is necessary for correct ground-truth label masking.

        """
        probe = "Probe am I"
        sentence1 = self.tokenizer.encode(conversation_lib.default_conversation.roles[1] + ": " + probe,
                                          bos=False, eos=False)
        sentence2 = self.tokenizer.encode(probe,
                                          bos=False, eos=False)
        if sentence1[-len(sentence2):] == sentence2:
            self.space_before_to_predict = False
        else:
            sentence3 = self.tokenizer.encode(" " + probe,
                                              bos=False, eos=False)
            assert sentence1[-len(sentence3):] == sentence3
            self.space_before_to_predict = True

    def add_speaker_and_signal(self, source, get_conversation=True):
        """Add speaker and start/end signal on each round."""
        BEGIN_SIGNAL = "### "
        END_SIGNAL = "\n"
        conversation = self.header

        to_predict_list = []

        for sentence in source:
            from_str = sentence["from"]
            if from_str.lower() in ["human"]:
                from_str = conversation_lib.default_conversation.roles[0]
            elif from_str.lower() in ["gpt", "assistant"]:
                from_str = conversation_lib.default_conversation.roles[1]
            else:
                raise ValueError(f"unknown dialog role: {from_str.lower()}")

            value = sentence["value"]
            if DEFAULT_IMAGE_TOKEN in value:
                value = value.replace(DEFAULT_IMAGE_TOKEN, '').strip()

            sentence_value = BEGIN_SIGNAL + from_str + ": " + value + END_SIGNAL

            if from_str == conversation_lib.default_conversation.roles[1]:
                to_predict_value = value + END_SIGNAL + "###"
                if self.space_before_to_predict:
                    to_predict_value = " " + to_predict_value
                to_predict_list.append(to_predict_value)

            if get_conversation:
                conversation = conversation + sentence_value

        conversation = conversation + BEGIN_SIGNAL
        return conversation, to_predict_list

cap_path = "./datasets/textual_annotations/mmfi/mmficap/mmfi_train_cs_full.json"
vqa_base_path = "./datasets/textual_annotations/mmfi/mmfivqa/train_cs_full/"

DATASETS = dict(

    mmfi_video=[
        dict(path=cap_path, type='mmfi_video'),
        dict(path=vqa_base_path + "mmfivqa_rgb.json", type='mmfi_video'),

    ],

    mmfi_depth=[
        dict(path=cap_path, type='mmfi_depth'),
        dict(path=vqa_base_path + "mmfivqa_depth.json", type='mmfi_depth'),
        # xrf55 video here
    ],


    mmfi_lidar=[
        dict(path=cap_path, type='mmfi_lidar'),
        dict(path=vqa_base_path + "mmfivqa_lidar.json", type='mmfi_lidar'),
        # xrf55 video here
    ],

    mmfi_mmwave=[
        dict(path=cap_path, type='mmfi_mmwave'),
        dict(path=vqa_base_path + "mmfivqa_mmwave.json", type='mmfi_mmwave'),
        # xrf55 video here
    ],

    mmfi_wifi=[
        dict(path=cap_path, type='mmfi_wifi'),
        dict(path=vqa_base_path + "mmfivqa_wifi.json", type='mmfi_wifi'),
        # xrf55 video here
    ]

)

class OneStageSenseDataset(Dataset):
    def __init__(self, dataset=['video'], transform=T_random_resized_crop, max_words=2048, image_words=30, tokenizer_path=None):
        if isinstance(dataset, str):
            dataset = [dataset]
        self.dataset = dataset
        # here for dataset loading - zch
        # test
        # dataset = ['xrf55_infra', 'xrf55_depth']
        group_ann = {}
        for d in dataset:
            for meta in DATASETS[d]:
                meta_path, meta_type = meta['path'], meta['type']
                meta_ext = os.path.splitext(meta_path)[-1]
                if meta_ext == ".json":
                    with open(meta_path) as f:
                        meta_l = json.load(f)
                        # add data_type
                        # this is a temp solution
                        new_meta_l = []
                        for l in meta_l:
                            l['data_type'] = meta_type
                            new_meta_l.append(l)
                        meta_l = new_meta_l
                elif meta_ext == ".jsonl":
                    meta_l = []
                    with open(meta_path) as f:
                        for i, line in enumerate(f):
                            try:
                                meta_l.append(json.loads(line))
                            except json.decoder.JSONDecodeError as e:
                                print(
                                    f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}", force=True)
                                raise e
                else:
                    raise NotImplementedError(
                        f"Unknown meta file extension: \"{meta_ext}\". "
                        f"Currently, .json, .jsonl are supported. "
                        "If you are using a supported format, please set the file extension so that the proper parsing "
                        "routine can be called."
                    )
                if meta_type not in group_ann:
                    group_ann[meta_type] = []
                print(f"{meta_path}, type {meta_type}: len {len(meta_l)}")
                group_ann[meta_type] += meta_l

        # sort group_ann for higher efficiency (items in one global batch with similar length)
        for meta_type, meta_l in group_ann.items():
            meta_l.sort(key=lambda data_item: sum(
                [len(_['value']) for _ in data_item['conversations']]))

        self.group_ann = group_ann
        self.ann = sum(list(self.group_ann.values()), start=[])

        self.group_indices = {}
        start_pos = 0
        for meta_type, meta_l in self.group_ann.items():
            self.group_indices[meta_type] = list(
                range(start_pos, start_pos + len(meta_l)))
            start_pos = start_pos + len(meta_l)

        print(f"total length: {len(self)}")
        self.transform = transform
        print(f"transform:\n{self.transform}")
        self.max_words = max_words
        self.image_words = image_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        self.conversation_generator = ConversationGenerator(self.tokenizer)

        
        self.load_funcs = dict(
            mmfi_video=self.load_mmfi_video,
            mmfi_depth=self.load_mmfi_depth,
            mmfi_infra=self.load_mmfi_infra,
            mmfi_lidar=self.load_mmfi_lidar,
            mmfi_mmwave=self.load_mmfi_mmwave,
            mmfi_wifi=self.load_mmfi_wifi,

            xrf55_video=self.load_xrf55_video,
            xrf55_depth=self.load_xrf55_depth,
            xrf55_infra=self.load_xrf55_infra,
            xrf55_wifi=self.load_xrf55_wifi,
            xrf55_rfid=self.load_xrf55_rfid,
        )
       


    def __len__(self):
        return len(self.ann)

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

    def load_mmfi_depth(self, data):

        depth_transform = transforms.Compose(
            [
                transforms.Resize(224),
            ]
        )

        # video_frame_transform = transforms.Compose(
        #     [
        #         transforms.Resize(224),
        #         transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # Normalize using ImageNet stats
        #     ]
        # )

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
            frame_tensor = torch.from_numpy(frame)
            # treat depth image to rgb image by repeat
            # all_depth.append(frame_tensor.unsqueeze(dim=0).repeat(3,1,1).unsqueeze(dim=1))  # [480, 640] -> [3, 1, 480, 640], 1 for temporal dim
            all_depth.append(frame_tensor.unsqueeze(dim=0).unsqueeze(dim=0))

        all_depth = [depth_transform(frame) for frame in all_depth]
        all_depth = video_utils.SpatialCrop(224, num_crops=3)(all_depth)
        all_depth = torch.stack(all_depth, dim=0)

        # [15, 1, 1, 224, 224] -> [15, 1, 224, 224], similar to load_video, depth only have one channel
        return all_depth[:,:,0]   
    
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
            frame_tensor = torch.from_numpy(frame)
            all_infra.append(frame_tensor)

        all_infra = torch.stack(all_infra, dim=0)
        # [5, 17, 2]  5 frames, each frame contain the keypoints from the infrared images.
        # what the difference between infra1 and infra2? They seems to be all (17,2) points.
        return all_infra

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
            frame_tensor = torch.from_numpy(data_tmp)
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
            frame_tensor = torch.from_numpy(data_tmp)
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
            frame_tensor = torch.from_numpy(data_frame)

            all_wifi.append(frame_tensor)
        
        all_wifi = torch.stack(all_wifi, dim=0)
        return all_wifi



    def load_xrf55_video(self, data):
 
        to_tensor_transform = transforms.ToTensor()
        # video_transform = transforms.Compose(
        #     [
        #         pv_transforms.ShortSideScale(224),
        #         NormalizeVideo(
        #             mean=(0.48145466, 0.4578275, 0.40821073),
        #             std=(0.26862954, 0.26130258, 0.27577711),
        #         ),
        #     ]
        # )

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
            all_video.append(frame_tensor)
            # all_video.append(frame_tensor.unsqueeze(dim=1))  # [3, 480, 640] -> [3, 1, 480, 640], 1 for temporal dim
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
            frame_tensor = torch.from_numpy(frame)
            all_depth.append(frame_tensor.unsqueeze(dim=0).unsqueeze(dim=0))

        all_depth = [depth_transform(frame) for frame in all_depth]
        while len(all_depth) < 10:
            all_depth.append(all_depth[-1])
        all_depth = video_utils.SpatialCrop(224, num_crops=3)(all_depth)
        all_depth = torch.stack(all_depth, dim=0)

        return all_depth[:,:,0] 

    def load_xrf55_wifi(self, data):
        video_path = data['video_path']
        if ".npy" in video_path:
            video_path = video_path.replace("Color", "WiFi")[0:-1]
        else: # cap
            video_path = video_path.replace("Color", "WiFi")[0:-1] + ".npy"

        all_wifi = np.load(video_path)
        frame_tensor = torch.from_numpy(all_wifi)
        frame_tensor = frame_tensor.view(9, 30, 5, 200).permute(2, 0, 1, 3)
        # all_wifi = frame_tensor.type(torch.float32) # [5, 9, 30, 200]
        all_wifi = frame_tensor
        return all_wifi


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
        # dtype = torch.float32
        return all_infra[:,:,0]   # [30, 1, 1, 224, 224] -> [30, 1, 224, 224]
    
    def load_xrf55_rfid(self, data):
        video_path = data['video_path']
        if ".npy" in video_path:
            video_path = video_path.replace("Color", "RFID")[0:-1]
        else: # cap
            video_path = video_path.replace("Color", "RFID")[0:-1] + ".npy"

        rfid_data = np.load(video_path)
        all_rfid = torch.from_numpy(rfid_data)
        return all_rfid

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


    def __getitem__(self, index, expect_type=None):
        if expect_type is None:
            data_item = self.ann[index]
        else:
            # in case we want get data from specific data_type
            data_item = self.group_ann[expect_type][index]

        data_type = data_item['data_type']
        if data_type != 'text':
            if data_type in self.load_funcs:
                try:
                    image = self.load_funcs[data_type](data_item)
                    if image == None:
                        raise ValueError('Data is None')
                except:
                    print('Error', data_item)
                    rand_idx = random.randint(
                        0, len(self.group_ann[data_type]))
                    return self.__getitem__(rand_idx, expect_type=data_type)
            else:
                raise ValueError(f'Does not support {data_type}')
        else:
            image = None
            # warnings.warn("pure black image for examples without image")
            # image = torch.zeros(3, 224, 224)

        source = data_item["conversations"]
        conversation, to_predict_values = self.conversation_generator.add_speaker_and_signal(
            source)
        if len(to_predict_values) == 0:
            warnings.warn(
                f"see dialog data with nothing to predict, data: {data_item}")
            return self[index-1]

        tokenzed_conversation = self.tokenizer.encode(
            conversation, bos=True, eos=True)
        labels = [IGNORE_INDEX for _ in tokenzed_conversation]
        
        # for mmfi
        action_label = int(data_item["video_path"].split("/")[-1][1:]) - 1  # label is [0-26]

        # for xrf55
        # video_path_split = data_item['video_path'].split("/")
        # action_label = int(video_path_split[-2].split("_")[1]) - 1
        
        check_pos = 0
        for value in to_predict_values:
            tokenized_value = self.tokenizer.encode(
                value, bos=False, eos=False)
            value_pos = find_sublist(
                tokenzed_conversation[check_pos:], tokenized_value) + check_pos
            if value_pos == -1:
                print(
                    "a sentence mismatches the corresponding piece in the conversation")
                return self[index-1]
            labels[value_pos:value_pos+len(tokenized_value)] = tokenized_value
            assert labels[value_pos:value_pos+len(
                tokenized_value)] == tokenzed_conversation[value_pos:value_pos+len(tokenized_value)]
            check_pos = value_pos+len(tokenized_value)

        input2 = torch.tensor(tokenzed_conversation, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        action_label = torch.tensor(action_label, dtype=torch.int64)

        if image is not None:
            max_words = self.max_words - self.image_words
        else:
            max_words = self.max_words
        padding = max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat(
                (input2, torch.zeros(padding, dtype=torch.int64) - 1))
            labels = torch.cat(
                (labels, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:max_words]
            labels = labels[:max_words]

        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        
       
        if image is None:
            return input2, labels, data_item['data_type']
        else:
            return input2, labels, action_label, image, data_item['data_type']

    def groups(self):
        return list(self.group_indices.values())


def find_sublist(a: list, b: list):
    len_a, len_b = len(a), len(b)
    for i in range(len_a - len_b + 1):
        if a[i:i+len_b] == b:
            return i
    return -1


class OneStageDistSampler(Sampler):
    #   Distrubuted Sampler ensuring data in a batch are of the same type (e.g. text, image-text)
    def __init__(self, dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, batch_size = None, acc_grad=1) -> None:
        if num_replicas is None or rank is None or rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid num_replicas ({num_replicas}) or rank ({rank})")
        assert batch_size is not None
        self.batch_size = batch_size

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.acc_grad = acc_grad
        self.epoch = 0
        self.start_iter = 0

        group_indices = dataset.groups()
        global_bsz = batch_size * num_replicas * acc_grad
        len_groups = [len(_) // global_bsz * global_bsz for _ in group_indices]
        group_indices = [indices[:len_indices] for indices, len_indices in zip(group_indices, len_groups)]
        group_n_batch = [len(_)//batch_size for _ in group_indices]
        assert all([_%num_replicas==0 for _ in group_n_batch])
        n_total_batch = sum(group_n_batch)

        assert n_total_batch % self.num_replicas == 0

        self.group_indices = group_indices

        self.total_size = n_total_batch * batch_size
        self.num_samples = self.total_size // num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        global_batch_size = self.batch_size * self.num_replicas * self.acc_grad
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            # self.group_indices should not be changed during shuffle. Only change copy.
            group_indices_shuffle = copy.deepcopy(self.group_indices)
            # for _ in group_indices_shuffle:
            #     rng.shuffle(_)
            global_batched_indices = [
                indices_in_group[i:i+global_batch_size]
                for indices_in_group in group_indices_shuffle
                for i in range(0, len(indices_in_group), global_batch_size)]
            rng.shuffle(global_batched_indices)
            indices = [_ for batch_indices in global_batched_indices for _ in batch_indices]
        else:
            group_indices = copy.deepcopy(self.group_indices)
            indices = [_ for batch_indices in group_indices for _ in batch_indices]

        assert len(indices) == self.total_size

        own_indices = []
        for start_pos in range(self.rank * self.batch_size, len(indices), self.num_replicas * self.batch_size):
            own_indices += indices[start_pos: start_pos + self.batch_size]
        # subsample
        assert len(own_indices) == self.num_samples

        if self.start_iter * self.batch_size > len(own_indices):
            own_indices = []
        else:
            own_indices = own_indices[self.start_iter * self.batch_size:]

        return iter(own_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int, start_iter: int = 0) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
            start_iter (int): start iter number.
        """
        self.epoch = epoch
        self.start_iter = start_iter

if __name__ == "__main__":

    pass

