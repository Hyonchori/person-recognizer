# Tracker for person re-identification

import os

import torch
import torchvision.transforms as T
import numpy as np
import cv2

from person import Person
from tracker_utils.metrics import cosine_similarity, cosine_similarity2


class Tracker():
    def __init__(self, query_root, feature_extractor, transform, device, img_size, top_k=10, cs_thr=0.4):
        self.feature_extractor = feature_extractor
        self.query_ids = [x for x in sorted(os.listdir(query_root)) if os.path.isdir(os.path.join(query_root, x))]
        self.query_paths = [os.path.join(query_root, x) for x in self.query_ids]

        self.transform = transform
        self.device = device
        self.top_k = top_k
        self.cs_thr = cs_thr
        self.person_queries = {query_id: Person(query_id, query_path,
                                                self.feature_extractor, self.transform, self.device, img_size)
                               for (query_id, query_path) in zip(self.query_ids, self.query_paths)}

    def cal_cosine_similarity(self, input_feat_batch):  # Calculate cosine similarity between input_feat and query_feat
        tmp = {idx: [] for idx in range(input_feat_batch.shape[0])}
        batch_shape = input_feat_batch.shape


        for person_id, person in self.person_queries.items():
            print(f"--------- ID {person_id} -----------")
            for input_feat in input_feat_batch:
                print("")
                input_feat_norm = input_feat / torch.linalg.norm(input_feat)
                query_feat_norm = person.query_feat_batch / torch.linalg.norm(person.query_feat_batch)
                dist_mat = cosine_similarity2(input_feat_norm[None], query_feat_norm).cpu().numpy()
                indices = np.argsort(dist_mat)[::-1][..., :self.top_k]
                top_dist = np.sort(dist_mat)[::-1][..., :self.top_k]
                top_mean = np.mean(top_dist)
                print(top_mean)
                print(top_dist)
                print(indices)
                print(person.query_img_batch.shape)
                if top_mean >= self.cs_thr:
                    cv2.imshow(f"matched with {person_id}", person.query_img_batch[indices[0]].numpy().transpose(1, 2, 0)[..., ::-1])

    def cal_cosine_similarity2(self, input_feat):
        id = -1
        print("")
        tmp = {}
        for person_id, person in self.person_queries.items():
            input_feat_norm = input_feat / torch.linalg.norm(input_feat)
            query_feat_norm = person.query_feat_batch / torch.linalg.norm(person.query_feat_batch)
            dist_mat = cosine_similarity2(input_feat_norm[None], query_feat_norm).cpu().numpy()
            indices = np.argsort(dist_mat)[::-1][..., :self.top_k]
            top_dist = np.sort(dist_mat)[::-1][..., :self.top_k]
            top_mean = np.mean(top_dist)
            print(top_mean)
            if top_mean >= self.cs_thr:
                cv2.imshow(f"matched with {person_id}",
                           person.query_img_batch[indices[0]].numpy().transpose(1, 2, 0)[..., ::-1])
                tmp[person_id] = top_mean
        if tmp:
            id = sorted(tmp.items(), key=lambda items: items[1], reverse=True)[0][0]
        return int(id)

    def plot_pca(self):
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, random_state=42)
        total_query = []
        total_id = []
        for person_id, person in self.person_queries.items():
            tmp_feats = person.query_feat_batch.cpu().numpy()
            total_query.append(tmp_feats)
            total_id += [int(person_id)] * tmp_feats.shape[0]
        total_query = np.vstack(total_query)

        query_dr = tsne.fit_transform(total_query)

        import matplotlib.pyplot as plt
        #plt.figure(figsize=(13, 10))
        plt.scatter(query_dr[:, 0], query_dr[:, 1], c=total_id)
        print(query_dr.shape)
        #plt.show()
        plt.savefig("save.png")
