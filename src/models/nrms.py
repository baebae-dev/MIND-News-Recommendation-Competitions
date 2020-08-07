import torch
import torch.nn as nn

from models.utils import SelfAttn, LinearAttn


class NRMS(nn.Module):
    def __init__(self, config, word2vec_embedding):
        super(NRMS, self).__init__()
        self.word_emb = nn.Embedding(word2vec_embedding.shape[0], config['word_emb_dim'])
        self.word_emb.weight = nn.Parameter(torch.tensor(word2vec_embedding, dtype=torch.float32))
        self.word_emb.weight.requires_grad = True
        self.news_dim = config['head_num'] * config['head_dim']
        self.user_dim = self.news_dim
        self.key_dim = config['attention_hidden_dim']

        self.news_encoder = NewsEncoder(word_emb=self.word_emb,
                                        drop=config['dropout'],
                                        word_dim=config['word_emb_dim'],
                                        news_dim=self.news_dim,
                                        key_dim=self.key_dim,
                                        head_num=config['head_num'],
                                        head_dim=config['head_dim'])
        self.user_encoder = UserEncoder(news_encoder=self.news_encoder,
                                        drop=config['dropout'],
                                        news_dim=self.news_dim,
                                        user_dim=self.user_dim,
                                        key_dim=self.key_dim,
                                        head_num=config['head_num'],
                                        head_dim=config['head_dim'])

    def forward(self, x, source):
        if source == 'history':
            his_out = self.user_encoder(x)
            return his_out
        elif source == 'candidate':
            cand_out = self.news_encoder(x)
            return cand_out


class Encoder(nn.Module):
    def __init__(self, drop, input_dim, output_dim, key_dim, head_num, head_dim):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p=drop)
        self.self_attn = SelfAttn(head_num=head_num,
                                  head_dim=head_dim,
                                  input_dim=input_dim)
        self.linear_attn = LinearAttn(output_dim=output_dim, key_dim=key_dim)

    def forward(self, **kwargs):
        pass


class NewsEncoder(Encoder):
    def __init__(self, word_emb, drop, word_dim, news_dim, key_dim, head_num, head_dim):
        super(NewsEncoder, self).__init__(drop=drop, input_dim=word_dim,
                                          output_dim=news_dim, key_dim=key_dim,
                                          head_num=head_num, head_dim=head_dim)
        self.word_emb = word_emb

    def forward(self, x):
        x = self.word_emb(x)
        out = self.dropout(x)
        out = self.self_attn(QKV=(out, out, out))
        out = self.dropout(out)
        out = self.linear_attn(out)
        return out


class UserEncoder(Encoder):
    def __init__(self, news_encoder, drop, news_dim, user_dim, key_dim, head_num, head_dim):
        super(UserEncoder, self).__init__(drop=drop, input_dim=news_dim,
                                          output_dim=user_dim, key_dim=key_dim,
                                          head_num=head_num, head_dim=head_dim)
        self.news_encoder = news_encoder

    def forward(self, x):
        x = self.news_encoder(x)
        out = self.dropout(x)
        out = self.self_attn(QKV=(out, out, out))
        out = self.dropout(out)
        out = self.linear_attn(out)
        return out
