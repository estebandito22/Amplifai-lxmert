# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 100
FAST_IMG_NUM = 500

# The path to data and image features.
POST_ACTION_IMGFEAT_ROOT = '/Volumes/My Passport/Amplifai/data_twitter/tweets/all_top_accounts_twitter_imgfeats_rand/'
POST_ACTION_METADATA_ROOT = '/Volumes/My Passport/Amplifai/data_metadata/post_action'
SPLIT2NAME = {
    'train': 'train',
    'val': 'val',
    'test': 'test',
}


class PostActionDataset:
    """
    A Post Action data example in json file:
        {
            "post_id": "728318860710051840",
            "label": {
                "announcement": 1
            },
            "sent": "First Unplugged Fall Tour Dates Announced! Stay tuned for more! Photo by Bill Bernstein\n<URL>"
        }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(os.path.join(POST_ACTION_METADATA_ROOT, "%s.json" % split))))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['post_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/post_action/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/post_action/trainval_label2ans.json"))
        self.label2ans = {int(k): v for k, v in self.label2ans.items()}
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""


class PostActionTorchDataset(Dataset):
    def __init__(self, dataset: PostActionDataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []
        # for split in dataset.splits:
        for split in ['train']:
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
            load_topk = 100 if (split == 'minival' and topk is None) else topk
            img_data.extend(load_obj_tsv(
                os.path.join(POST_ACTION_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                topk=load_topk,
                id2datum=self.raw_dataset.id2datum))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['post_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        post_id = datum['post_id']
        sent = datum['sent']

        # Get image info
        img_info = self.imgid2img[post_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return post_id, feats, boxes, sent, target
        else:
            return post_id, feats, boxes, sent


class PostActionEvaluator:
    def __init__(self, dataset: PostActionDataset):
        self.dataset = dataset

    def evaluate(self, postid2ans: dict):
        score = 0.
        for postid, ans in postid2ans.items():
            datum = self.dataset.id2datum[postid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(postid2ans)

    def dump_result(self, postid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "post_id": int,
                "answer": str
            }

        :param postid2ans: dict of postid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for post_id, ans in postid2ans.items():
                result.append({
                    'post_id': post_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


