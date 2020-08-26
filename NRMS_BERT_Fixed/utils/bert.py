import pandas as pd
import torch
import itertools
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import pickle
import os, sys

def get_title_embedding_bert(df, tokenizer, model, pickle_file, title_size=30, n=30):
    nid2index = {}
    titles = ['']

    for r in tqdm(range(df.shape[0])):
        nid = df[0][r]
        title = df[3][r]

        if nid in nid2index:
            continue
        nid2index[nid] = len(nid2index) + 1
        titles.append('[CLS]' + title + '[SEP]')

    tokenized_titles = tokenizer(titles, max_length =title_size, padding=True, truncation=True,
                                 add_special_tokens=False, return_tensors='pt')

    def get_tokenized_titles(t, _from, _to):
        res = {}
        for key in t:
            res[key] = t[key][_from:_to]
        return res

    num_news = len(titles)
    gap = num_news // (n-1)

    ranges = [i * gap for i in range(n)]
    ranges.append(num_news)
    for i in tqdm(range(n)):
        news_title_embeds = (model(**get_tokenized_titles(tokenized_titles, ranges[i], ranges[i+1]))[0])
        news_title_embeds = news_title_embeds.half()
        with open(pickle_file + f'{i}', 'wb') as f:
            pickle.dump(news_title_embeds, f)

    for i in range(n):
        file_name = pickle_file + f'{i}'
        with open(file_name, 'rb') as f:
            if i == 0:
                z = pickle.load(f)
            else:
                z = torch.cat((z, pickle.load(f)))
        os.remove(file_name) # remove used file
        res_file = [nid2index, z]
    with open(pickle_file, 'wb') as f:
        pickle.dump(res_file, f, protocol=4)


def get_category_embedding_bert(df, tokenizer, model, pickle_file):
    cats = df[1]
    def get_cat_embedding(cat):
        return model(**tokenizer(cat, return_tensors='pt'))[0].squeeze()[1].half()
    cats = cats.map(lambda x: get_cat_embedding(x))

    with open(pickle_file, 'wb') as f:
        pickle.dump(cats, f, protocol=4)


def get_subcategory_embedding_bert(df, tokenizer, model, pickle_file):

    subcats = df[2]
    def get_cat_embedding(subcat):
        return model(**tokenizer(subcat, return_tensors='pt'))[0].squeeze()[1].half()
    subcats = subcats.map(lambda x: get_cat_embedding(x))

    with open(pickle_file, 'wb') as f:
        pickle.dump(subcats, f, protocol=4)

if __name__ == "__main__":
    print('start')
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # [demo, large], ['train', 'dev']
    dt1, dt2 = 'demo', 'train'
    for dt1 in ['demo', 'large']:
        for dt2 in ['train', 'dev']:
            news_file = f'/data/mind/MIND{dt1}_{dt2}/news.tsv'
            df = pd.read_csv(news_file, sep='\t', header=None)

            # get title embedding
            title_size = 30
            title_pickle_file = f'/data/mind/MIND{dt1}_{dt2}/BERT/bert_title_{title_size}.pickle'
            # get_title_embedding_bert(df, tokenizer, model, title_pickle_file, title_size=30, n=30)
            print(f"{dt1}-{dt2}/ get title embedding done")

            # get category embedding
            category_pickle_file = f'/data/mind/MIND{dt1}_{dt2}/BERT/bert_category.pickle'
            get_category_embedding_bert(df, tokenizer, model, category_pickle_file)
            print(f"{dt1}-{dt2}/ get category embedding done")

            # get subcategory embedding
            subcategory_pickle_file = f'/data/mind/MIND{dt1}_{dt2}/BERT/bert_subcategory.pickle'
            get_subcategory_embedding_bert(df, tokenizer, model, category_pickle_file)
            print(f"{dt1}-{dt2}/ get subcategory embedding done")

            # checks
            with open(title_pickle_file, 'rb') as tf:
                nid2idx, title_embeddings = pickle.load(tf)
            with open(category_pickle_file, 'rb') as cf:
                cats = pickle.load(cf)
            with open(subcategory_pickle_file, 'rb') as sf:
                subcats = pickle.load(cf)
            print(nid2idx.shape)
            print(cats.shape)
            print(subcats.shape)


            print(f"{dt1}-{dt2} done")

