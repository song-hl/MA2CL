import torch
import torch.nn as nn

from .util import init

"""CNN Modules and utils."""


class CNNLayer(nn.Module):
    def __init__(
        self,
        obs_shape,
        hidden_size,
        use_orthogonal=True,
        use_ReLU=True,
        kernel_size=3,
        stride=1,
    ):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])
        self.need_normalize = obs_shape[1] > 32

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        if obs_shape[1] < 32:
            cnn = nn.Sequential(
                init_(nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=1)),
                active_func,
                init_(nn.Conv2d(32, 64, kernel_size=1, stride=1)),
                active_func,
                init_(
                    nn.Conv2d(64, hidden_size, kernel_size=kernel_size, stride=stride)
                ),
                active_func,
                nn.Flatten(),
            )
        else:
            cnn = nn.Sequential(
                init_(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                active_func,
                init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                active_func,
                init_(
                    nn.Conv2d(64, hidden_size, kernel_size=kernel_size, stride=stride)
                ),
                active_func,
                nn.Flatten(),
            )
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            n_flatten = cnn(torch.as_tensor(dummy_input).float()).shape[1]
        self.net = nn.Sequential(
            cnn, init_(nn.Linear(n_flatten, hidden_size)), active_func,
        )
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x / (255.0 if self.need_normalize else 1.0)
        x = self.net(x)
        x = self.ln(x)
        return x


class CNNBase(nn.Module):
    def __init__(self, args, obs_shape):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size

        self.cnn = CNNLayer(
            obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU
        )
        # self.cnn = ImpalaCNN(obs_shape, self.hidden_size)

    def forward(self, x):
        x = self.cnn(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
        )
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class ImpalaCNN(nn.Module):
    """
    Network from IMPALA paper implemented in ModelV2 API.
    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_shape, out_features=64):
        super().__init__()
        c, h, w = obs_shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(
            in_features=shape[0] * shape[1] * shape[2], out_features=out_features
        )

    def forward(self, x):
        x = x / 255.0  # scale to 0-1
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.hidden_fc(x)
        x = nn.functional.relu(x)
        return x
