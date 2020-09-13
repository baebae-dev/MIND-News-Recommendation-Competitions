######################################################################################################
# mind2020
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: TANR/src/model/general/click_predictor/dot_product.py
# - model general dot_product
#
# Version: 1.0
#######################################################################################################

import torch


class DotProductClickPredictor(torch.nn.Module):
    def __init__(self):
        super(DotProductClickPredictor, self).__init__()

    def forward(self, candidate_news_vector, user_vector):
        """
        Args:
            candidate_news_vector: batch_size, X
            user_vector: batch_size, X
        Returns:
            (shape): batch_size
        """
        # batch_size
        probability = torch.bmm(
            user_vector.unsqueeze(dim=1),
            candidate_news_vector.unsqueeze(dim=2)).flatten()
        return probability
