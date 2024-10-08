import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BinaryLinear(torch.nn.Module):
    def __init__(self, in_shape, out_shape, kernel_size=3):
        super(BinaryLinear, self).__init__()
        self.shape = (out_shape, in_shape)
        self.weight = torch.nn.Parameter((torch.rand((self.shape)) * 2 - 1) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(abs(real_weights))
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        # binary_weights_no_grad = torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.linear(x, binary_weights)

        return y

class BHDC(torch.nn.Module):
    def __init__(self, inshape = 10000, outshape = 10, dropout_prob = 0):
        super(BHDC, self).__init__()

        self.inshape = inshape
        self.outshape = outshape
        self.dropout_prob = dropout_prob

        self.binary_weight_l2 = BinaryLinear(inshape, outshape)
        self.dropout = torch.nn.Dropout(p = dropout_prob)

    def forward(self, x):
        x = x.view(-1, self.inshape)        # 将输入数据展平为一维向量，大小为 (batch_size, inshape)
        x = self.dropout(x)
        x = self.binary_weight_l2(x)
        return x
