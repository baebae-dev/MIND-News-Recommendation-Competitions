import os
import pickle
import pandas as pd

class NewsRecommender:
    # popular bucket (clicked): [key/ value] = [bucket index/ list of news]
    # popular bucket (recommended): [key/ value] = [bucket index/ list of news]
    # sorted list of newses in order of time. (Only containing new keys)

    def __init__(self, data_type):
        # data_type: 'dev', 'train'
        self.bucket_size = 3 # size of buckets

        # Define file names
        self.data_path = '/datia/mind/' + 'MINDlarge_' + data_type + '/'
        self.behavior_file = self.data_path + 'behaviors.tsv'
        self.news_file = self.data_path + 'news.tsv' # combined version

        self.fresh_news_df_file = self.data_path + "fresh_news_df.pickle"
        self.pop_hash_clicked_file = self.data_path + "pop_hash_clicked_{}.pickle".format(self.bucket_size)
        self.pop_hash_recommended_file = self.data_path + "pop_hash_recommended_{}.pickle".format(self.bucket_size)

        # Popular news and sorted news list
        self.pop_hash_clicked = None # [key/ value] = [bucket index/ list of news]
        self.pop_hash_recommended = None # [key/ value] = [bucket index/ list of news]
        self.fresh_news_df = None # sorted list of news ids


    def get_fresh_news_df(self, time, k):
        # input:  time - query time
        #         k - number of news
        # output: list of fresh news IDs

        # Load or generate sorted newslist file
        if self.fresh_news_df == None:
            self.load_fresh_news_df()

        # get list of fresh news
        _df = self.fresh_news_df[self.fresh_news_df['Date'] >= self._process_date(time)]
        _df.sort_values(by='Date', ascending=True, inplace=True)
        return _df[0:k,1].values



    def get_pop_news_clicked(self, time, k):
        if self.pop_hash_clicked == None:
            self.load_pop_hash_clicked()
        ltime = self.parsing_time(time)
        bucket_key = (ltime[0], ltime[1], ltime[2], ltime[3]//self.bucket_size)
        sorted_news_list = sorted(self.pop_hash_clicked[bucket_key].items(),
                                  reverse=True, key=lambda item: item[1])
        news_list = sorted_news_list[0:k]
        return news_list


    def get_pop_news_recommended(self, time, k):
        if self.pop_hash_recommended == None:
            self.load_pop_hash_recommended()
        ltime = self.parsing_time(time)
        bucket_key = (ltime[0], ltime[1], ltime[2], ltime[3] // self.bucket_size)
        sorted_news_list = sorted(self.pop_hash_recommended[bucket_key].items(),
                                  reverse=True, key=lambda item: item[1])
        news_list = sorted_news_list[0:k]
        return news_list

    def load_fresh_news_df(self):
        if os.path.isfile(self.fresh_news_df_file):
            with open(self.fresh_news_df_file, 'rb') as f:
                self.fresh_news_df = pickle.load(f)
        else:
            self.generate_fresh_news_df()

    def load_pop_hash_clicked(self):
        if os.path.isfile(self.pop_hash_clicked_file):
            with open(self.pop_hash_clicked_file, 'rb') as f:
                self.pop_hash_clicked = pickle.load(f)
        else:
            self.generate_pop_hash_clicked()

    def load_pop_hash_recommended(self):
        if os.path.isfile(self.pop_hash_recommended_file):
            with open(self.pop_hash_recommended_file, 'rb') as f:
                self.pop_hash_recommended = pickle.load(f)
        else:
            self.generate_pop_hash_recommended()

    def generate_fresh_news_df(self):
        news_list = []
        _df = pd.read_csv(self.news_file)[['Date', '0']]

        _df['Date'] = _df['Date'].map(self._process_date)
        _df.sort_values(by='Date', ascending=True, inplace=True)

        self.fresh_news_df = _df
        with open(self.fresh_news_df_file, 'wb') as f:
            pickle.dump(_df, f)

    def generate_pop_hash_clicked(self):
        df = pd.read_csv(self.behavior_file, sep='\t', header=None)
        buckets = {}
        for i in range(df.shape[0]):
            _imp = df.iloc[i,:]
            ltime = self.parsing_time(_imp[2])
            clicked_imps = self.get_news_list(_imp[4], all=False)
            bucket_key = (ltime[0], ltime[1], ltime[2], ltime[3]//self.bucket_size)

            if bucket_key in buckets.keys():
                _d = buckets[bucket_key]
                for _imp in clicked_imps:
                    if _imp in _d.keys():
                        _d[_imp] += 1
                    else:
                        _d[_imp] = 1
            else:
                buckets[bucket_key] = {}
                for _imp in clicked_imps:
                    buckets[bucket_key][_imp] = 1
        # Save bucket as a pickle file
        with open(self.pop_hash_clicked_file, 'wb') as f:
            pickle.dump(buckets, f)
        self.pop_hash_clicked = buckets

    def generate_pop_hash_recommended(self):
        df = pd.read_csv(self.behavior_file, sep='\t', header=None)
        buckets = {}
        for i in range(df.shape[0]):
            _imp = df.iloc[i, :]
            ltime = self.parsing_time(_imp[2])
            recommended_imps = self.get_news_list(_imp[4], all=True)
            bucket_key = (ltime[0], ltime[1], ltime[2], ltime[3] // self.bucket_size)

            if bucket_key in buckets.keys():
                _d = buckets[bucket_key]
                for _imp in recommended_imps:
                    if _imp in _d.keys():
                        _d[_imp] += 1
                    else:
                        _d[_imp] = 1
            else:
                buckets[bucket_key] = {}
                for _imp in recommended_imps:
                    buckets[bucket_key][_imp] = 1

        # Save bucket as a pickle file
        with open(self.pop_hash_recommended_file, 'wb') as f:
            pickle.dump(buckets, f)

        self.pop_hash_recommended = buckets

    def parsing_time(self, ftime):
        _mdy = ftime.split(' ')[0].split('/')
        _hmt = ftime.split(' ')[1].split(':')
        if ftime.split(' ')[2] == 'PM':
            _hmt[0] = int(_hmt[0])
            _hmt[0] += 12
        ltime = [int(_mdy[0]), int(_mdy[1]), int(_mdy[2]),
                 int(_hmt[0]), int(_hmt[1]), int(_hmt[2])]
        return ltime

    def get_news_list(self, raw_imp, all=False):
        _imps = raw_imp.split(' ')
        if all:
            _news_list = [_i[:-2] for _i in _imps]
        else:
            _news_list = [_i[:-2] for _i in _imps if _i[-1] == '1']
        return _news_list

    def _process_date(self, date):
        if type(date) == float:
            return 1e15
        t1 = date.split('T')[0].split('-')
        t2 = date.split('T')[1][:-1].split(':')
        res = t1[0] + t1[1] + t1[2] + t2[0] + t2[1] + t2[2]
        return int(res)

if __name__ == "__main__":
    nr = NewsRecommender('dev')
    ns = nr.get_pop_news_recommended('11/15/2019 8:55:22 AM', k=10)
    print(ns)
