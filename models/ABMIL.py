# reuse implementation from AdditiveMIL paper.
import torch
import torch.nn as nn
from typeguard import typechecked
from typing import Tuple, Optional, Sequence
from torch.nn import functional as F


# from .common import Sum, StableSoftmax

class StableSoftmax(torch.nn.Module):
    @typechecked
    def __init__(self, dim=0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.nn.LogSoftmax(dim=self.dim)(inputs).exp()


class Sum:
    def pool(self, features, **kwargs) -> int:
        return torch.sum(features, **kwargs)


class DefaultAttentionModule(torch.nn.Module):
    @typechecked
    def __init__(
            self,
            input_dims: int,
            hidden_dims: Sequence[int] = (),
            hidden_activation: torch.nn.Module = nn.ReLU(),
            output_activation: torch.nn.Module = StableSoftmax(dim=1),
            use_batch_norm: bool = True,
            track_bn_stats: bool = True,
    ):

        super().__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_batch_norm = use_batch_norm
        self.track_bn_stats = track_bn_stats

        self.model = self.build_model()

    def build_model(self):
        nodes_by_layer = [self.input_dims] + list(self.hidden_dims) + [1]
        layers = []
        iterable = enumerate(zip(nodes_by_layer[:-1], nodes_by_layer[1:]))
        for i, (nodes_in, nodes_out) in iterable:
            layer = nn.Linear(in_features=nodes_in, out_features=nodes_out, bias=True)
            layers.append(layer)
            if i < len(self.hidden_dims):
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(nodes_out, track_running_stats=self.track_bn_stats))
                layers.append(self.hidden_activation)
        model = nn.Sequential(*layers)
        return model

    def forward(self, features, bag_size):
        out = self.model(features)
        out = out.view([-1, bag_size])
        attention = self.output_activation(out)
        return attention.unsqueeze(-1)


class DefaultClassifier(torch.nn.Module):
    @typechecked
    def __init__(
            self,
            input_dims: int,
            output_dims: int,
            hidden_dims: Sequence[int] = (),
            hidden_activation: torch.nn.Module = torch.nn.ReLU(),
            pooling_mode=Sum(),
    ):

        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.pooling_mode = pooling_mode
        self.model = self.build_model()

    def build_model(self):
        nodes_by_layer = [self.input_dims] + list(self.hidden_dims) + [self.output_dims]
        layers = []
        iterable = enumerate(zip(nodes_by_layer[:-1], nodes_by_layer[1:]))
        for i, (nodes_in, nodes_out) in iterable:
            layer = torch.nn.Linear(in_features=nodes_in, out_features=nodes_out)
            layers.append(layer)
            if i < len(self.hidden_dims):
                layers.append(self.hidden_activation)
        model = torch.nn.Sequential(*layers)
        return model

    def forward(self, features, attention):
        features = self.pooling_mode.pool(attention * features, dim=1, keepdim=False)
        logits = self.model(features)
        classifier_out_dict = {}
        classifier_out_dict['logits'] = logits
        return classifier_out_dict


class DefaultMILGraph(torch.nn.Module):
    @typechecked
    def __init__(self, input_dims=512, output_dims=2):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.pointer = DefaultAttentionModule(hidden_activation=nn.LeakyReLU(0.2), hidden_dims=[256, 256],
                                              input_dims=input_dims)
        self.classifier = DefaultClassifier(hidden_dims=[256, 256], input_dims=input_dims, output_dims=output_dims)

    def forward(self, wsi_features):
        ## features: [batch_size, bag_size, feature_dim]
        features = wsi_features
        batch_size = features.shape[0]  # batch_size:1
        bag_size = features.shape[1]  # bag_size:6000
        features = features.squeeze()
        attention = self.pointer(features, bag_size)  # attention: [batch_size, bag_size, 1]

        if not torch.all(attention >= 0):
            raise ValueError("{}: Attention weights cannot be negative".format(attention))

        features = features.view([batch_size, bag_size] + list(features.shape[1:]))  # separate batch and bag dim
        classifier_out_dict = self.classifier(features, attention)
        patch_logits = classifier_out_dict['patch_logits'] if 'patch_logits' in classifier_out_dict else None

        out = {}
        bag_logits = classifier_out_dict['logits']
        Y_hat = torch.argmax(bag_logits, dim=1)
        Y_prob = F.softmax(bag_logits, dim=1)
        out['bag_logits'] = bag_logits
        out['Y_prob'] = Y_prob
        out['Y_hat'] = Y_hat
        if patch_logits is not None:
            out['patch_logits'] = patch_logits
        out['attention'] = attention

        return out


if __name__ == "__main__":
    data = torch.randn((1, 6000, 512)).cuda()
    model = DefaultMILGraph(input_dims=512, output_dims=2).cuda()
    print(model.eval())
    results_dict = model(wsi_features=data)
    print('bag_logits: ', results_dict['bag_logits'])  # torch.Size([1, 2])
    print('Y_prob: ', results_dict['Y_prob'])  # torch.Size([1, 2])
    print('Y_hat: ', results_dict['Y_hat'])  # torch.Size([1])
    print('attention: ', results_dict['attention'].shape)  # torch.Size([1, 6000, 1])

    '''
    bag_logits:  tensor([[-0.0312,  0.0419]], device='cuda:0', grad_fn=<AddmmBackward0>)
    Y_prob:  tensor([[0.4817, 0.5183]], device='cuda:0', grad_fn=<SoftmaxBackward0>)
    Y_hat:  tensor([1], device='cuda:0')
    attention:  torch.Size([1, 6000, 1])


    '''