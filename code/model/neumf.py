import torch
import torch.nn as nn
import re


class NeuMF(nn.Module):
    @staticmethod
    def __create_layer(in_size, dropout_prob=0.5):
        out_size = in_size // 2
        return nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ), out_size

    @staticmethod
    def load(name, state_dict, dropout_prob=0.5):
        user_number = 0
        item_number = 0
        predictive_factor = 0
        mlp_layer = 0

        regex = r'^mlp_layer_X\.(\d+)\.1\.weight$'
        for layer in state_dict:
            match = re.search(regex, layer)
            if match:
                layer = int(match.group(1)) + 1
                if layer > mlp_layer:
                    mlp_layer = layer
            elif layer == 'mlp_P.weight':
                user_number = state_dict[layer].shape[0]
            elif layer == 'mlp_Q.weight':
                item_number = state_dict[layer].shape[0]
            elif layer == 'neu_mf.weight':
                predictive_factor = state_dict[layer].shape[1]

        model = NeuMF(
            name, user_number, item_number,
            predictive_factor_num=predictive_factor,
            mlp_layer_num=mlp_layer,
            dropout_prob=dropout_prob
        )
        model.load_state_dict(state_dict)
        return model

    def __init__(self, name, user_number, item_number, predictive_factor_num, mlp_layer_num, dropout_prob=0.5):
        super(NeuMF, self).__init__()

        self.name = name

        self.user_number = user_number
        self.item_number = item_number

        assert predictive_factor_num % 2 == 0, "Number of predictive factor should be divisible by 2."

        # MLP
        assert mlp_layer_num >= 1, "MLP should have at least one layer."

        mlp_out_size = predictive_factor_num // 2
        mlp_in_size = ((2 ** mlp_layer_num) * mlp_out_size)

        self.mlp_embedding_dim = mlp_in_size // 2
        self.mlp_P = nn.Embedding(num_embeddings=self.user_number, embedding_dim=self.mlp_embedding_dim)
        self.mlp_Q = nn.Embedding(num_embeddings=self.item_number, embedding_dim=self.mlp_embedding_dim)

        last_out_size = mlp_in_size
        layers = []
        for i in range(mlp_layer_num):
            layer, last_out_size = NeuMF.__create_layer(last_out_size, dropout_prob)
            layers.append(layer)
        self.mlp_layer_X = nn.Sequential(*layers)
        # END OF MLP

        # GMF
        self.gmf_embedding_dim = predictive_factor_num // 2
        self.gmf_P = nn.Embedding(num_embeddings=self.user_number, embedding_dim=self.gmf_embedding_dim)
        self.gmf_Q = nn.Embedding(num_embeddings=self.item_number, embedding_dim=self.gmf_embedding_dim)
        # END OF GMF

        self.neu_mf = nn.Linear(predictive_factor_num, 1)

        self.__init_weights()

    def __init_weights(self):
        nn.init.normal_(self.gmf_P.weight, std=.01)
        nn.init.normal_(self.gmf_Q.weight, std=.01)

        nn.init.normal_(self.mlp_P.weight, std=.01)
        nn.init.normal_(self.mlp_Q.weight, std=.01)

        for mlp_layer in self.mlp_layer_X:
            for inner_layer in mlp_layer:
                if isinstance(inner_layer, nn.Linear):
                    nn.init.xavier_uniform_(inner_layer.weight)
                    inner_layer.bias.data.zero_()

        nn.init.kaiming_uniform_(self.neu_mf.weight, nonlinearity="sigmoid")
        self.neu_mf.bias.data.zero_()

    def forward(self, user_id, item_id):
        user_index = torch.sub(user_id, 1)
        item_index = torch.sub(item_id, 1)

        # GMF
        gfm_p_u = self.gmf_P(user_index)
        gfm_q_i = self.gmf_Q(item_index)
        gfm_out = torch.multiply(gfm_p_u, gfm_q_i)

        # MLP
        mlp_p_u = self.mlp_P(user_index)
        mlp_q_i = self.mlp_Q(item_index)
        layer_1_out = torch.cat((mlp_p_u, mlp_q_i), -1)
        mlp_out = self.mlp_layer_X(layer_1_out)

        neu_cf_in = torch.cat((gfm_out, mlp_out), -1)
        return self.neu_mf(neu_cf_in)
