######################################################################################################
# mind2020
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: TANR/src/config.py
# - Default Settings for Model and Learning
#
# Version: 1.0
#######################################################################################################

import os

model_name = os.environ['MODEL_NAME'] if 'MODEL_NAME' in os.environ else 'TANR'
# Currently included model
assert model_name == 'TANR'

class BaseConfig():
    """
    General configurations appiled to all models
    """
    num_batches = 60000  # Number of batches to train
    num_batches_show_loss = 100  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 800
    batch_size = 128
    learning_rate = 0.001
    validation_proportion = 0 # 0% -> no validation set and use dev dataset 
    num_workers = 4  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    num_words_title = 20
    num_words_abstract = 50
    word_freq_threshold = 3
    entity_freq_threshold = 3
    entity_confidence_threshold = 0.5
    negative_sampling_ratio = 4  # K
    dropout_probability = 0.2
    # Modify the following by the output of `src/dataprocess.py`
    num_words = 1 + 44774
    num_categories = 1 + 295
    num_entities = 1 + 14697
    num_users = 1 + 711222
    word_embedding_dim = 300
    category_embedding_dim = 100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200


class TANRConfig(BaseConfig):
dataset_attributes = {"news": ['category', 'title'], "record": []}
# For CNN
num_filters = 300
window_size = 3
topic_classification_loss_weight = 0.1


