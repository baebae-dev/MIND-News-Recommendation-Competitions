######################################################################################################
# mind2020
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: TANR/src/evaluate_file.py
# - make submission file for test set
#
# Version: 1.0
#######################################################################################################

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from config import model_name
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from ast import literal_eval
import importlib
import pickle
import csv
import zipfile
 
try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except (AttributeError, ModuleNotFoundError):
    print(f"{model_name} not included!")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def value2rank(d):
    values = list(d.values())
    ranks = [sorted(values, reverse=True).index(x) for x in values]
    return {k: ranks[i] + 1 for i, k in enumerate(d.keys())}

 
class NewsDataset(Dataset):
    """
    Load news for evaluation.
    """
    def __init__(self, news_path):
        super(NewsDataset, self).__init__()
        self.news_parsed = pd.read_csv(news_path, sep='\t',
                                         converters={
                                             'title': literal_eval,
                                             'abstract': literal_eval,
                                             'title_entities': literal_eval,
                                             'abstract_entities': literal_eval
                                         }, quoting=csv.QUOTE_NONE)

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        row = self.news_parsed.iloc[idx]
        item = {
            "id": row.id,
            "category": row.category, 
            "subcategory": row.subcategory,
            "title": row.title,
            "abstract": row.abstract,
            "title_entities": row.title_entities,
            "abstract_entities": row.abstract_entities
        }
        return item


class UserDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    """
    def __init__(self, behaviors_path, user2int_path):
        super(UserDataset, self).__init__()
        self.behaviors =  pd.read_csv(behaviors_path, sep='\t',
                                       header=None,
                                       usecols=[1, 3],
                                       names=['user', 'clicked_news'], quoting=csv.QUOTE_NONE)

        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user in user2int:
                self.behaviors.at[row.Index, 'user'] = user2int[row.user]
            else:
                user_missed += 1
                self.behaviors.at[row.Index, 'user'] = 0
        

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user":
            row.user,
            "clicked_news_string":
            row.clicked_news,
            "clicked_news":
            row.clicked_news.split()[:config.num_clicked_news_a_user]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"].extend(['PADDED_NEWS'] * repeated_times)

        return item


class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """
    def __init__(self, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=range(5),
                                       names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "impression_id": row.impression_id,
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions
        }
        return item


@torch.no_grad()
def evaluate(model, directory, generate_txt=False, txt_path=None):
    """
    Evaluate model on target directory.
    Args:
        model: model to be evaluated
        directory: the directory that contains two files (behaviors.tsv, news_parsed.tsv)
        generate_txt: whether to generate txt file from inference result
        txt_path: file path
    Returns:
        scores.txt
        prediction.txt
        prediction.zip
    """
    news_dataset = NewsDataset(os.path.join(directory, 'news_parsed.tsv'))
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False)

    news2vector = {}
    with tqdm(total=len(news_dataloader),
              desc="Calculating vectors for news") as pbar:
        for minibatch in news_dataloader:
            news_ids = minibatch["id"]
            if any(id not in news2vector for id in news_ids):
                news_vector = model.get_news_vector(minibatch)
                for id, vector in zip(news_ids, news_vector):
                    if id not in news2vector:
                        news2vector[id] = vector
            pbar.update(1)

    news2vector['PADDED_NEWS'] = torch.zeros(
        list(news2vector.values())[0].size())

    user_dataset = UserDataset(os.path.join(directory, 'behaviors.tsv'),
                               '../data/train/user2int.tsv')
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False)

    user2vector = {}
    with tqdm(total=len(user_dataloader),
              desc="Calculating vectors for users") as pbar:
        for minibatch in user_dataloader:
            user_strings = minibatch["clicked_news_string"] 
            if any(user_string not in user2vector
                   for user_string in user_strings):

                clicked_news_vector = torch.stack([ 
                    torch.stack([news2vector[x].to(device) for x in news_list], dim=0)
                    for news_list in minibatch["clicked_news"]
                ],  dim=0).transpose(0, 1)
                
                user_vector = model.get_user_vector(clicked_news_vector)
                for user, vector in zip(user_strings, user_vector):
                    if user not in user2vector:
                        user2vector[user] = vector
            pbar.update(1)

    behaviors_dataset = BehaviorsDataset(
        os.path.join(directory, 'behaviors.tsv'))
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=config.num_workers)


    # file generate 
    if generate_txt:
        # scores.txt save
        scores_dict = dict()
        # prediction txt saves (rank)
        answer_file = open(txt_path, 'w')

    with tqdm(total=len(behaviors_dataloader),
              desc="Calculating probabilities") as pbar:
        for minibatch in behaviors_dataloader:

            impression = {
                news[0].split('-')[0]: model.get_prediction(
                    news2vector[news[0].split('-')[0]], 
                    user2vector[minibatch['clicked_news_string'][0]]).item()
                for news in minibatch['impressions']
            }

            y_pred_list = list(impression.values())

            if generate_txt:
                # scores saves
                scores_dict[minibatch['impression_id'][0]]  = y_pred_list
                # rank saves
                answer_file.write(
                    f"{minibatch['impression_id'][0]} {str(list(value2rank(impression).values())).replace(' ','')}\n"
                )
            pbar.update(1)

    if generate_txt:
        # scores save 
        with open('../data/test/scores.pickle', 'wb') as f:
            pickle.dump(scores_dict, f, protocol=4)
        # rank saves
        answer_file.close()

    return 


if __name__ == '__main__':
    print('Using device:', device)
    print(f'Evaluating model {model_name}')
    # Don't need to load pretrained word/entity/context embedding
    # since it will be loaded from checkpoint later
    model = Model(config).to(device)
    
    from train import latest_checkpoint  # Avoid circular imports
    checkpoint_dir = os.path.join('../checkpoint', model_name)
    checkpoint_path = latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        print('No checkpoint file found!')
        exit()   
    print(f"Load saved parameters in {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    evaluate(model, '../data/test', True, '../data/test/prediction.txt')
    print('save the file', '../data/test/scores.pickles')
    print('save the file', '../data/test/prediction.txt')
    
    # zip file save
    f = zipfile.ZipFile('../data/test/prediction.txt', 'w', zipfile.ZIP_DEFLATED)
    print('zip the file', '../data/test/prediction.txt')
    f.write('../data/test/prediction.txt', arcname='prediction.txt')
    f.close()

    print('zippped file ', '../data/test/prediction.zip')

 