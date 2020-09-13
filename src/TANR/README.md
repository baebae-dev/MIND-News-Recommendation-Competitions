######################################################################################################
# TRANSMETER
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: TANR/README.md
# - readme for run model TANR
#
# Version: 1.0
#######################################################################################################

# News Recommendation

| Model     | Full name                                                                 | Paper                                              |
| --------- | ------------------------------------------------------------------------- | -------------------------------------------------- |
| TANR      | Neural News Recommendation with Topic-Aware News Representation            | https://www.aclweb.org/anthology/P19-1110.pdf               |

## Get started

Basic setup.

```
pip3 install -r requirements.txt
```

Download and preprocess the data.

```bash
mkdir data && cd data
# Download GloVe pre-trained word embedding
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
sudo apt install unzip
unzip glove.840B.300d.zip -d glove
rm glove.840B.300d.zip

# Download MIND dataset
# By downloading the dataset, you agree to the [Microsoft Research License Terms](https://go.microsoft.com/fwlink/?LinkID=206977). For more detail about the dataset, see https://msnews.github.io/.
wget https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip
unzip MINDlarge_train.zip -d train
unzip MINDlarge_dev.zip -d test
rm MINDlarge_*.zip

# Preprocess data into appropriate format
cd ..
python3 src/data_preprocess.py
# Remember you shoud modify `num_*` in `src/config.py` by the output of `src/data_preprocess.py`

# Train model
cd src
python3 train.py

# evaluate for dev set
python3 evaluate.py

# make submission file for test set
python3 evaluate_file.py

```

Modify `src/config.py` to select target model. The configuration file is organized into general part (which is applied to all models) and model-specific part (that some models not have).

```bash
vim src/config.py
```

Run.

```bash
# Train and save checkpoint into `checkpoint/{model_name}/` directory
python3 src/train.py
# Load latest checkpoint and evaluate on the test set
# This will also generate prediction file `data/test/prediction.txt`
python3 src/evaluate.py

# or

chmod +x run.sh
./run.sh
```

