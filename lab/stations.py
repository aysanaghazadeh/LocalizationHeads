import os
import os.path as osp
import pickle
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

class DefaultsVars:
    def __init__(self):
        self.model_config = {
            "llm_name": "llama-v2-7b",
            "config": None
        }
        
        self.state = {
            "logic_flag": {},
            "save_to_path": {},
            "attn_weights": {},  # Store attention weights by layer
        }
        
        self.metadata = {
            "vis_len": 0,
            "qid": None,
            "output_token_count": -1,
            "current_decoder_layer": 0,
            "gt_label": None,
            "answer": None,
            "answer_ids": None,
            "prompt": None,
            "image_path": None,
        }
        
        self.segments = {
            "id_pieces": {
                "system": [], "role_0": [], "image": [],
                "inst_q": [], "role_1": []
            },
            "text_pieces": {
                "system": [], "role_0": [], "image": [],
                "inst_q": [], "role_1": []
            },
            "begin_pos": {
                "system": float("-inf"), "role_0": float("-inf"),
                "vis": float("-inf"), "inst_q": float("-inf"),
                "role_1": float("-inf")
            }
        }

class StationEngine:
    _defaults = DefaultsVars()
    model_config = _defaults.model_config
    state = _defaults.state
    metadata = _defaults.metadata
    segments = _defaults.segments

    @classmethod
    def activate(cls):
        cls.set_flag(True)
        print(f"{cls.__name__} flag set to {cls._flag()}")

    @classmethod
    def export_model_config(cls, config):
        cls.model_config = config

    @classmethod
    def set_flag(cls, flag_value=True):
        cls.state["logic_flag"][cls.__name__] = flag_value

    @classmethod
    def _flag(cls, name=None):
        if name is not None:
            return cls.state["logic_flag"].get(name, False)
        return cls.state["logic_flag"].get(cls.__name__, False)

    @classmethod
    def set_save_to_path(cls, save_to_path):
        cls.state["save_to_path"][cls.__name__] = save_to_path

    @classmethod
    def get_save_to_path(cls):
        return cls.state["save_to_path"].get(cls.__name__, None)

    @classmethod
    def run_logic(cls):
        raise NotImplementedError("Running basic logic in LogicEngine.")

    @classmethod
    def clear(cls):
        """Reset all class variables to their initial state"""
        defaults = DefaultsVars()
        defaults.state.update(
            {
                "logic_flag": cls.state["logic_flag"],
                "save_to_path": cls.state["save_to_path"],
            }
        )
        defaults.model_config = cls.model_config
        
        cls.state = defaults.state
        cls.model_config = defaults.model_config
        cls.metadata = defaults.metadata
        cls.segments = defaults.segments

    @classmethod
    def save_to_pickle(cls, path, data):
        if osp.exists(path):
            print(f"File already exists at {path}.")
            return
            
        with open(path, "wb") as f:
            pickle.dump(data, f)
            
    @classmethod 
    def detach_to_cpu(cls, tensor_or_dict):
        if isinstance(tensor_or_dict, torch.Tensor):
            return tensor_or_dict.detach().cpu()
        elif isinstance(tensor_or_dict, dict):
            return {k: cls.detach_to_cpu(v) for k,v in tensor_or_dict.items()}
        return tensor_or_dict


class ValueMonitor(StationEngine):
    @classmethod
    def remember_layer(cls, l):
        cls.__base__.metadata["current_decoder_layer"] = l

    @classmethod
    def count_output_token(cls):
        cls.__base__.metadata["output_token_count"] += 1

    @classmethod
    def remember_seg_token_order(cls, n):
        cls.__base__.seg_token_order.append(n)

    @classmethod
    def remember_qid(cls, qid):
        cls.__base__.metadata["qid"] = qid


class AttentionStation(StationEngine):
    
    @classmethod
    def store_attn(cls, attn, layer_idx):
        """Store attention weights for a specific layer"""
        if "attn_weights" not in cls.state:
            cls.state["attn_weights"] = {}
        # @seil: We consider a case where only the batch is 1
        cls.state["attn_weights"][layer_idx] = cls.detach_to_cpu(attn[0]) # @seil: reducing batch dimension
    
    @classmethod
    def get_attn_weights(cls):
        """Get all stored attention weights"""
        return cls.state.get("attn_weights", {})
    
    @classmethod
    def clear_attn_weights(cls):
        """Clear stored attention weights"""
        if "attn_weights" in cls.state:
            cls.state["attn_weights"] = {}
    
    @classmethod
    def save_attn(cls, save_to_path, attn=None):
        # Use stored attention weights if none provided
        if attn is None and cls.get_attn_weights():
            # Convert dictionary to list format expected by downstream code
            layers = sorted(cls.get_attn_weights().keys())
            if not layers:
                print("No attention weights stored")
                return
                
            # Format attention weights as expected by the localization head finder
            attn_list = []
            for layer_idx in layers:
                attn_list.append(cls.state["attn_weights"][layer_idx])
            attn = torch.stack(attn_list, dim=0)
        
        # Process and save attention weights
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().cpu()
        elif isinstance(attn, list):
            attn = [a.detach().cpu() if isinstance(a, torch.Tensor) and a.device.type == "cuda" else a for a in attn]
            # Handle nested lists of tensors
            if attn and isinstance(attn[0], list):
                for i, a_list in enumerate(attn):
                    attn[i] = [aa.detach().cpu() if isinstance(aa, torch.Tensor) and aa.device.type == "cuda" else aa for aa in a_list]

        os.makedirs(osp.dirname(save_to_path), exist_ok=True)
        with open(save_to_path, "wb") as f:
            pickle.dump(attn, f)


class MetadataStation(StationEngine):
    @classmethod
    def set_image_path(cls, image_path):
        cls.state["image_path"] = image_path

    @classmethod
    def set_qid(cls, qid):
        cls.state["qid"] = qid

    @classmethod
    def get_begin_pos(cls, key):
        return cls.segments.get("begin_pos").get(key)

    @classmethod
    def get_metadata(cls):
        """Returns the metadata of the class."""
        return {
            "prompt": cls.state.get("prompt"),
            "gt_label": cls.state.get("gt_label"),
            "answer": cls.state.get("answer"),
            "answer_ids": cls.state.get("answer_ids"),
            "id_pieces": cls.segments.get("id_pieces"),
            "text_pieces": cls.segments.get("text_pieces"),
            "begin_pos": cls.segments.get("begin_pos"),
            "vis_len": cls.state.get("vis_len"),
            "image_path": cls.state.get("image_path"),
        }

    @classmethod
    def get_vis_len(cls):
        return cls.metadata.get("vis_len")

    @classmethod
    def set_correct(cls, correct):
        cls.state["correct"] = correct

    @classmethod
    def set_prompt(cls, prompt):
        cls.state["prompt"] = prompt

    @classmethod
    def set_answer(cls, answer_ids, answer):
        if isinstance(answer_ids, torch.Tensor):
            answer_ids = answer_ids.cpu().clone()

        cls.state["answer_ids"] = answer_ids
        cls.state["answer"] = answer

    @classmethod
    def set_gt_label(cls, gt_label):
        cls.state["gt_label"] = gt_label

    @classmethod
    def set_vis_len(cls, vis_len):
        cls.metadata["vis_len"] = vis_len

    @classmethod
    def set_begin_pos(cls, key, idx):
        cls.segments["begin_pos"][key] = idx

    @classmethod
    def set_id_pieces(cls, key, ids: List[int]):
        assert isinstance(ids, list)
        cls.segments["id_pieces"][key] = ids

    @classmethod
    def set_text_pieces(cls, key, texts: List[str]):
        assert isinstance(texts, list)
        cls.segments["text_pieces"][key] = texts

    @classmethod
    def save_metadata(cls, save_to_path, **kwargs):
        if osp.exists(save_to_path):
            print(f"File already exists at {save_to_path}.")
            return
            
        metadata = cls.metadata.copy()
        metadata.update(kwargs)
        
        os.makedirs(osp.dirname(save_to_path), exist_ok=True)
        with open(save_to_path, "wb") as f:
            pickle.dump(metadata, f)

    @classmethod
    def prompt_segments(cls, input_ids, tokenizer, image_token_id, conv_template, image_token_idx=-200, use_mm_start_end=False):
        def tokenize_segment(text: str, add_special_tokens: bool = False) -> List[int]:
            return tokenizer(text, add_special_tokens=add_special_tokens).input_ids

        if use_mm_start_end:
            prompt = tokenizer.decode(input_ids[: image_token_id - 1]) + "<im_start><image><im_end>" + tokenizer.decode(input_ids[image_token_id + 2 :])
        else:
            prompt = tokenizer.decode(input_ids[:image_token_id]) + "<image>" + tokenizer.decode(input_ids[image_token_id + 1 :])
        
        segments = {}
        role_0, role_1 = conv_template.roles
        if role_0 == "" and role_1 == "": # conv: referseg
            sents_splits = prompt.split("<image>", 1)
            segments["system"] = sents_splits[0]
            segments["vis"] = "<image>".strip()
            segments["text"] = sents_splits[1]

            tokenized_segments = {
                "system": tokenize_segment(segments["system"], False),
                "vis": [image_token_idx] if segments["vis"] else [],
                "text": tokenize_segment(segments["text"], False),
            }

            merge_tokenized_segments = []
            for l in list(tokenized_segments.values()):
                merge_tokenized_segments.extend(l)

            # Check with original input id vs. tokenized segments
            for a, b in zip(input_ids, merge_tokenized_segments):
                assert a == b, f"Token mismatch: {a} != {b}"

            token_id_range = {
                "system": [tokenized_segments["system"][0], tokenized_segments["vis"][0]],
                "vis": [tokenized_segments["vis"][0], tokenized_segments["text"][0]],
                "text": [tokenized_segments["text"][0], -1],
            }
            
            # Filter whitespace tokens
            whitespace_token = 29871  # Whitespace token ID
            for key, value in tokenized_segments.items():
                tokenized_segments[key] = [v for v in value if v != whitespace_token]

            merge_tokenized_segments = []
            for l in list(tokenized_segments.values()):
                merge_tokenized_segments.extend(l)

            input_ids_renewed = [i for i in input_ids if i != whitespace_token]

        else:
            role_0 = ' '+role_0
            role_1 = ' '+role_1
            user_split = prompt.split(role_0, 1)
            if len(user_split) > 1:
                segments["system"] = user_split[0].rstrip()
                remaining_text = role_0 + user_split[1]
            else:
                segments["system"] = prompt
                remaining_text = ""

            assistant_split = remaining_text.split(role_1, 1)
            if len(assistant_split) > 1:
                user_text = assistant_split[0].rstrip()
                segments["role_1"] = role_1 + assistant_split[1]
            else:
                segments["role_1"] = ""
                user_text = remaining_text

            if "<image>" in user_text:
                user_text_part, inst_q_text = user_text.split("<image>", 1)
                segments["role_0"] = user_text_part
                segments["image"] = "<image>"
                segments["inst_q"] = inst_q_text
            else:
                segments["role_0"] = user_text
                segments["image"] = None
                segments["inst_q"] = ""

            tokenized_segments = {
                "system": tokenize_segment(segments["system"], False),
                "role_0": tokenize_segment(segments["role_0"], False),
                "image": [image_token_idx] if segments["image"] else [],
                "inst_q": tokenize_segment(segments["inst_q"], False),
                "role_1": tokenize_segment(segments["role_1"], False),
            }
            
            # Filter whitespace tokens
            whitespace_token = 29871  # Whitespace token ID
            for key, value in tokenized_segments.items():
                tokenized_segments[key] = [v for v in value if v != whitespace_token]

            merge_tokenized_segments = []
            for l in list(tokenized_segments.values()):
                merge_tokenized_segments.extend(l)

            input_ids_renewed = [i for i in input_ids if i != whitespace_token]

            # Check with original input id vs. tokenized segments
            for a, b in zip(input_ids_renewed, merge_tokenized_segments):
                assert a == b, f"Token mismatch: {a} != {b}"

            # Set metadata; id_pieces 
            for key, value in tokenized_segments.items():
                cls.set_id_pieces(key=key, ids=value)
            
            token_id_range = {
                "system": [tokenized_segments["system"][0], tokenized_segments["role_0"][0]],
                "role_0": [tokenized_segments["role_0"][0], tokenized_segments["vis"][0]],
                "vis": [tokenized_segments["vis"][0], tokenized_segments["inst_q"][0]],
                "inst_q": [tokenized_segments["inst_q"][0], tokenized_segments["role_1"][0]],
                "role_1": [tokenized_segments["role_1"][0], -1],
            }

        # Set metadata; begin_pos and text_pieces
        begin_pos = 0
        for key, token_ids in tokenized_segments.items():
            cls.set_begin_pos(key=key, idx=begin_pos)
            if key == "vis":
                end_pos = begin_pos + cls.metadata["vis_len"]
            else:
                end_pos = begin_pos + len(token_ids)
            begin_pos = end_pos
            cls.set_text_pieces(key=key, texts=[tokenizer.decode(token_ids)] if key != "vis" else ["<image>"])

        return token_id_range, input_ids_renewed


class VisionDataStation(StationEngine):
    @classmethod
    def set_after_ve(cls, after_ve):
        cls.__base__.state["after_ve"] = after_ve

    @classmethod
    def set_after_proj(cls, after_proj):
        cls.__base__.state["after_proj"] = after_proj

class IoUStation(StationEngine):
    path_gt_mask = None

    @classmethod
    def set_path_gt_mask(cls, path_gt_mask):
        cls.path_gt_mask = path_gt_mask

    @classmethod
    def _resize_attn(cls, vis_attn: torch.Tensor, hw: Tuple[int, int]):
        vis_attn = vis_attn.detach().cpu()
        vis_attn = F.interpolate(
            vis_attn.unsqueeze(0).unsqueeze(0), 
            size=hw, 
            mode='bilinear', 
            align_corners=False)[0,0]
        return vis_attn

    @classmethod
    def _compute_iou(cls, mask_gt, mask_pred):
        mask_gt = mask_gt.detach().cpu()
        mask_pred = mask_pred.detach().cpu()
        if not isinstance(mask_gt, torch.Tensor):
            mask_gt = torch.tensor(mask_gt, dtype=torch.bool)
        if not isinstance(mask_pred, torch.Tensor):
            mask_pred = torch.tensor(mask_pred, dtype=torch.bool)
        
        mask_gt = mask_gt.flatten().bool()
        mask_pred = (mask_pred > mask_pred.mean()).flatten().bool()

        intersection = (mask_gt & mask_pred).sum().item()
        union = (mask_gt | mask_pred).sum().item()

        iou = intersection / (union + 1e-6)
        return iou
    
    @classmethod
    def compute_iou(cls, attn, name_mask_png):
        """
        attn : List[Tensor[L, H, T, T]]
        """
        mask_path = osp.join(cls.path_gt_mask, name_mask_png)
        if not osp.exists(mask_path):
            print(f"Mask file not found: {mask_path}")
            return None
            
        mask_gt = T.ToTensor()(Image.open(mask_path)) # [1, H, W]
        iou_head_wise = torch.zeros((attn.shape[0], attn.shape[2], 1), dtype=torch.float32) # [L, H, 1]
        vis_len = MetadataStation.metadata["vis_len"]
        begin_im = MetadataStation.segments["begin_pos"]["image"]
        end_im = begin_im + vis_len
        idx_tok = MetadataStation.segments["begin_pos"]["role_1"] + vis_len - 1
        
        for l in range(attn.shape[0]):
            for h in range(attn.shape[2]):
                vis_attn = cls._resize_attn(attn[l, 0, h, idx_tok, begin_im:end_im][None, :], mask_gt.shape[-2:]) # [1, H, W]
                iou = cls._compute_iou(mask_gt, vis_attn)
                iou_head_wise[l, h] = iou
                
        flat_index = torch.argmax(iou_head_wise).item()
        l, h, c = np.unravel_index(flat_index, iou_head_wise.shape)
        print(f"MAX IoU: {iou_head_wise.max().item():.4f}, Layer: {l}, Head: {h}")
        return iou_head_wise
    
    @classmethod
    def save_results(cls, attn, line, path_save):
        """
        attn : List[Tensor[L, H, T, T]]
        line : Dict with image path
        """
        name_mask_png = osp.splitext(osp.basename(line["image"]))[0] + ".png"
        iou = cls.compute_iou(attn, name_mask_png) # [H, 1]
        if iou is not None:
            iou = cls.detach_to_cpu(iou)
            cls.save_to_pickle(path_save, iou)