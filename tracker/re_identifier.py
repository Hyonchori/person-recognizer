# Tracker for person re-identification

import os

import torch
import torchvision.transforms as T
import numpy as np
import cv2

from person import Person
from tracker_utils.metrics import cosine_similarity, cosine_similarity2


class Tracker():
    def __init__(self, query_root, feature_extractor, transform, device, img_size, top_k=10):
        self.feature_extractor = feature_extractor
        self.query_ids = [x for x in sorted(os.listdir(query_root)) if os.path.isdir(os.path.join(query_root, x))]
        self.query_paths = [os.path.join(query_root, x) for x in self.query_ids]

        self.transform = transform
        self.device = device
        self.top_k = top_k
        self.person_queries = {query_id: Person(query_id, query_path,
                                                self.feature_extractor, self.transform, self.device, img_size)
                               for (query_id, query_path) in zip(self.query_ids, self.query_paths)}

    def cal_cs(self, input_feat_batch):  # Calculate cosine similarity between input_feat and query_feat
        tmp = {idx: [] for idx in range(input_feat_batch.shape[0])}
        batch_shape = input_feat_batch.shape

        for person_id, person in self.person_queries.items():
            for input_feat in input_feat_batch:
                dist_mat = cosine_similarity2(input_feat[None], person.query_feat_batch).cpu().numpy()
                indices = np.argsort(dist_mat)[..., :self.top_k]
                top_dist = np.sort(dist_mat)[..., :self.top_k]
                top_mean = np.mean(top_dist)
                print(top_mean)