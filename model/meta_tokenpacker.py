from typing import List
import torch
import torch.nn as nn
import json
import os
import math
from .tokenizer import Tokenizer
from . import LLM

from fairscale.nn.model_parallel import initialize as fs_init

# import sys
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
import json

class MetaTokenpackerModel(nn.Module):

    def __init__(self, llama_type, llama_config, llama_ckpt_dir=None, tokenizer_path=None):
        super().__init__()
        
    
        self.ckpt_path = "/home/LLM_ckpt/tokenpacker/"
        tokenpacker_config_json = self.ckpt_path + "config.json"

        with open(tokenpacker_config_json, 'r') as file:
            tokenpacker_config_dict = json.load(file)

        tokenpacker_config = LlamaConfig.from_dict(tokenpacker_config_dict)
        # loss
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.criterion_action = torch.nn.CrossEntropyLoss()
        
        ModelArgs = LLM.__dict__[llama_type].ModelArgs
        Transformer = LLM.__dict__[llama_type].Transformer  # onellm, could be changed to other type, e.g., sensingllm

        with open(llama_config, "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=2048, max_batch_size=32, **params
        )
        self.tokenizer = Tokenizer(model_path=tokenizer_path)  
        model_args.vocab_size = self.tokenizer.n_words

        
        model = Transformer(tokenpacker_config, model_args)  # the clip is loaded in Transformer by open_clip
        
        mp_rank = fs_init.get_model_parallel_rank()
        if llama_ckpt_dir is not None:
            ckpt_path = os.path.join(llama_ckpt_dir, f"consolidated.{mp_rank:02d}.pth")  
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                msg = model.load_state_dict(checkpoint, strict=False) # load llama, no clip is reasonable
                print(msg)
            else:
                print(f'Checkpoint not found at {ckpt_path}')
        self.llma = model
        
        
        for name, param in self.named_parameters():
            if param.requires_grad:
               print(f"Trainable param: {name}, {param.shape}, {param.dtype}")
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameter count : {count}")

    def forward(self, examples, labels, action_label, image=None, modal='image'):
        output, output_action = self.llma(examples, image=image, modal=modal)  # here is the main model of OneLLM, llama type:onellm
        # The predicted token at time t (output[:, :-1, :]).
        # he true label at time t (labels[:, 1:]).
        # for a sequence [START, a, b, c, d, END]
        # model can not predict START (no prefix), can just predict following based on START.
        # output = [a, b, c, d, END, NONE]
        # labels = [START, a, b, c, d, END] thus use this operation to align [a, b, c, d, END]
        output = output[:, :-1, :] 
        labels = labels[:, 1:]

        if labels.sum() == 0:
            c_loss = output.mean() * 0
            a_loss = output_action.mean() * 0
        else:
        
            c_loss = self.criterion(output.reshape(-1, 32000), labels.flatten())
   
            if output_action.shape[-1] == 27:
                a_tmp = output_action.reshape(-1, 27)
            else:
                a_tmp = output_action.reshape(-1, 55)
            if torch.isnan(a_tmp).any():
                print("error")
            if output_action.shape[-1] == 27:
                a_loss = self.criterion_action(output_action.reshape(-1, 27), action_label.flatten()) 
            else:
                a_loss = self.criterion_action(output_action.reshape(-1, 55), action_label.flatten()) 
            # a_loss = self.criterion_action(output_action.reshape(-1, 55), action_label.flatten()) 
            # a_loss_value = a_loss.item()
            # if not math.isfinite(a_loss_value):
            #     print("[Rank {}] a_loss is {}, stopping training".format(dist.get_rank(), a_loss_value), force=True)

        return c_loss, a_loss

    def generate(
        self,
        prompts: List[str],
        images,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        modal = ['image'],
    ) -> List[str]:
        bsz = len(prompts)
        params = self.llma.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full(
            (bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0


        past_key_values = None  # use for new kv cache

        for cur_pos in range(start_pos, total_len):
            logits, past_key_values = self.llma.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos, images if prev_pos == 0 else None, modal=modal, past_key_values=past_key_values)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                try:
                    next_token = self.sample_top_p(probs, top_p)
                except Exception as e:  # inf in probs
                    # next_token = self.sample_top_p(probs, top_p)
                    next_token = torch.zeros((16,1)).cuda()
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

            past_key_values = past_key_values  

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded

    def predict_action(
        self,
        prompts: List[str],
        images,
        modal = ['video'],
    ):
        bsz = len(prompts)
        params = self.llma.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_prompt_size)

        tokens = torch.full(
            (bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        cur_pos = start_pos
        
        images = images.squeeze()  
        if len(images.shape) != 5:  # video [32, 15, 3, 224, 224]
            images = images.unsqueeze(dim=2)  # [32, 15, 1, 224, 224]
        images = images.half()  # changed to half
        logits = self.llma.forward_action_inference(tokens[:, prev_pos:cur_pos], prev_pos, images if prev_pos == 0 else None, modal=modal)
        
        return logits

    def extract_features(
        self,
        prompts: List[str],
        images,
        modal = ['mmfi_video'],
    ):

        bsz = len(prompts)
        params = self.llma.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_prompt_size)

        tokens = torch.full(
            (bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        cur_pos = start_pos

        images = images.squeeze()  
        if len(images.shape) != 5:  # video [32, 15, 3, 224, 224]
            images = images.unsqueeze(dim=2)  # [32, 15, 1, 224, 224]
        images = images.half()  # changed to half
        before_connector_feats, after_connector_feats, llm_image_feats, h_caption = self.llma.forward_extract_features(tokens[:, prev_pos:cur_pos], images, modal=modal)
        
        return before_connector_feats, after_connector_feats, llm_image_feats, h_caption
    
    @torch.inference_mode()
    def stream_generate(
        self,
        prompt: str,
        images,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        modal = ['image'],
    ):
        params = self.llma.params

        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        # truncate from the left. leave some space for generation.
        max_seq_len = params.max_seq_len
        if images is not None:
            max_seq_len -= self.llma.image_words

        max_prompt_size = max_seq_len - max_gen_len
        prompt_tokens = prompt_tokens[-max_prompt_size:]

        prompt_size = len(prompt_tokens)

        total_len = min(max_seq_len, max_gen_len + prompt_size)

        tokens = torch.full([total_len], 0).cuda().long()

        tokens[:len(prompt_tokens)] = torch.tensor(prompt_tokens).long()
        start_pos = prompt_size
        prev_pos = 0
        generate_until = start_pos
        for cur_pos in range(start_pos, total_len):
            logits = self.llma.forward_inference(tokens[None, prev_pos:cur_pos], prev_pos, images if prev_pos == 0 else None, modal = modal)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.item()

            if next_token == self.tokenizer.eos_id:
                break

            tokens[cur_pos] = next_token
            prev_pos = cur_pos
            generate_until = cur_pos + 1
            yield {"text": self.tokenizer.decode(tokens[start_pos:generate_until].tolist()), "end_of_content": False}

        yield {"text": self.tokenizer.decode(tokens[start_pos:generate_until].tolist()), "end_of_content": True}

    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def get_image_words(self):
        return self.llma.image_words