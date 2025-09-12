import torch
import torch.nn.functional as F
import torch.nn as nn


class Sum:
    def pool(self, features, **kwargs) -> int:
        return torch.sum(features, **kwargs)


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A


class ProtoMIL(nn.Module):
    def __init__(self, input_dims, output_dims, pooling_mode=Sum()):  # K, L, N
        super(ProtoMIL, self).__init__()
        self.output_class = output_dims
        self.attn = Attn_Net_Gated(L=input_dims)
        self.classifier = nn.Linear(input_dims, output_dims, bias=True)
        self.pooling_mode = pooling_mode

    def forward(self, wsi_features):  # N x K, N x C
        wsi_features = wsi_features.squeeze(0)  # N x K
        attn = self.attn(wsi_features)  # N x 1
        # compute attention of deep features, A_deep [N, 1]
        attn = F.softmax(attn, 0)  # N x 1
        # softmax the attention of deep features, seems like the sum of all attention values for patches will be 1. attention raw score becomes probability.
        weighted_feat = attn * wsi_features  # Final output (B) - NxD
        pooled_feat = self.pooling_mode.pool(weighted_feat, dim=0, keepdim=True)  ### sum pooling at the bag dimension.
        # apply attention on projected deep features
        logits = self.classifier(pooled_feat)  # N x 1
        # perform classification on the attention applied deep features
        bag_logits = logits
        Y_hat = torch.argmax(bag_logits, dim=1)
        Y_prob = F.softmax(bag_logits, dim=1)
        out = {}
        out['bag_logits'] = bag_logits
        out['Y_prob'] = Y_prob
        out['Y_hat'] = Y_hat
        out['attn'] = attn ## attention values.
        return out

if __name__ == "__main__":
    data = torch.randn((1, 6000, 512)).cuda()
    model = ProtoMIL(input_dims=512, output_dims=2).cuda()
    out = model(wsi_features=data)
    print(out)


