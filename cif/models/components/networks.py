import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parents[3] / "gitmodules" / "nsf"))
try:
    from nn import ConvEncoder, ConvDecoder
finally:
    sys.path.pop(0)


class ConstantNetwork(nn.Module):
    def __init__(self, value, fixed):
        super().__init__()
        if fixed:
            self.register_buffer("value", value)
        else:
            self.value = nn.Parameter(value)

    def forward(self, inputs):
        return self.value.expand(inputs.shape[0], *self.value.shape)


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv1 = self._get_conv3x3(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv2 = self._get_conv3x3(num_channels)

    def forward(self, inputs):
        out = self.bn1(inputs)
        out = torch.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)

        out = out + inputs

        return out

    def _get_conv3x3(self, num_channels):
        return nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,

            # We don't add a bias here since any subsequent ResidualBlock
            # will begin with a batch norm. However, we add a bias at the
            # output of the whole network.
            bias=False
        )


class ScaledTanh2dModule(nn.Module):
    def __init__(self, module, num_channels):
        super().__init__()
        self.module = module
        self.weights = nn.Parameter(torch.ones(num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_channels, 1, 1))

    def forward(self, inputs):
        out = self.module(inputs)
        out = self.weights * torch.tanh(out) + self.bias
        return out


def get_resnet(
        num_input_channels,
        hidden_channels,
        num_output_channels
):
    num_hidden_channels = hidden_channels[0] if hidden_channels else num_output_channels

    # TODO: Should we have an input batch norm?
    layers = [
        nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=num_hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
    ]

    for num_hidden_channels in hidden_channels:
        layers.append(ResidualBlock(num_hidden_channels))

    layers += [
        nn.BatchNorm2d(num_hidden_channels),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=num_hidden_channels,
            out_channels=num_output_channels,
            kernel_size=1,
            padding=0,
            bias=True
        )
    ]

    # TODO: Should we have an output batch norm?

    return ScaledTanh2dModule(
        module=nn.Sequential(*layers),
        num_channels=num_output_channels
    )


def get_mlp(
        num_input_channels,
        hidden_channels,
        num_output_channels,
        activation,
        log_softmax_outputs=False
):
    layers = []
    prev_num_hidden_channels = num_input_channels
    for num_hidden_channels in hidden_channels:
        layers.append(nn.Linear(prev_num_hidden_channels, num_hidden_channels))
        layers.append(activation())
        prev_num_hidden_channels = num_hidden_channels
    layers.append(nn.Linear(prev_num_hidden_channels, num_output_channels))

    if log_softmax_outputs:
        layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


class VAEOneLayerDecoder(nn.Module):
    def __init__(self, latent_dim, data_shape, num_hidden_channels, kernel_size, stride):
        super().__init__()
        self.fc = nn.Linear(latent_dim, int(np.prod(data_shape[1:])*num_hidden_channels/stride**2))
        self.act = nn.Tanh()

        pre_convt_size = [int(d / stride) for d in data_shape[1:]]
        self.reshape = lambda x: x.view(-1, num_hidden_channels, *pre_convt_size)
        self.convt = nn.ConvTranspose2d(num_hidden_channels, data_shape[0], kernel_size, stride, padding=1)

    def forward(self, x):
        x = self.act(self.fc(x))
        x = self.convt(self.reshape(x))
        return x


class OneLayerCNN(nn.Module):
    def __init__(
            self,
            input_shape,
            output_dim,
            num_hidden_channels,
            kernel_size,
            stride,
            activation
    ):
        super().__init__()

        # XXX: Does not robustly select amount of padding - may end up with wrong sizes
        self.conv = nn.Conv2d(input_shape[0], num_hidden_channels, kernel_size, stride, padding=1)
        self.act = activation()

        linear_input_shape = int(np.prod(input_shape[1:]) * num_hidden_channels / stride**2)
        self.reshape = lambda x: x.view(-1, linear_input_shape)
        self.fc = nn.Linear(linear_input_shape, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.reshape(x)
        x = self.fc(x)

        return x


class TupleMLP(nn.Module):
    def __init__(
            self,
            input_shapes,
            hidden_channels,
            num_output_channels,
            activation
    ):
        super().__init__()

        # XXX: input_shapes is of the form ((n_z,), (n_x,))
        num_input_channels = input_shapes[0][0] + input_shapes[1][0]
        self.mlp = get_mlp(num_input_channels, hidden_channels, num_output_channels, activation)

    def forward(self, tuple_inputs):
        inputs = torch.cat(tuple_inputs, dim=1)
        return self.mlp(inputs)


class TupleConvnetUpscaleVector(nn.Module):
    def __init__(
            self,
            input_shapes,
            hidden_channels,
            num_output_channels,
            activation,
            up_scale=4,
            kernel_size=4,
            stride=2
    ):
        super().__init__()

        # XXX: input_shapes is of the form ((n_z,), (c_x,d_x,d_x))
        image_channels = input_shapes[1][0]
        image_dim = input_shapes[1][1]
        pre_upsample_dim = image_dim // up_scale

        self.fc_latent = nn.Linear(input_shapes[0][0], pre_upsample_dim**2)
        self.reshape_latent = (lambda x:
            x.view(-1, 1, pre_upsample_dim, pre_upsample_dim)
        )
        self.upsampler = nn.Upsample(scale_factor=up_scale, mode="bilinear")

        self.convnet = OneLayerCNN(
            input_shape=(image_channels+1, image_dim, image_dim),
            output_dim=num_output_channels,
            num_hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation
        )

    def forward(self, tuple_inputs):
        latents, data = tuple_inputs
        latents_as_image = self.upsampler(
            self.reshape_latent(
                self.fc_latent(latents)
            )
        )
        new_data = torch.cat((latents_as_image, data), dim=1)

        outputs = self.convnet(new_data)
        return outputs


class TupleResnetVectorizeImage(nn.Module):
    def __init__(
            self,
            input_shapes,
            hidden_channels,
            num_output_channels,
            activation
    ):
        super().__init__()

        # XXX: input_shapes is of the form ((n_z,), (c_x,d_x,d_x))
        image_channels = input_shapes[1][0]
        image_dim = input_shapes[1][1]
        latent_dim = input_shapes[0][0]

        resnet_hidden_channels = hidden_channels[:-1]
        fc_hidden_channels = hidden_channels[-1]

        self.resnet = get_resnet(
            num_input_channels=image_channels,
            hidden_channels=resnet_hidden_channels,
            num_output_channels=1
        )
        self.activation = activation()

        self.mlp = get_mlp(
            num_input_channels=latent_dim+image_dim**2,
            hidden_channels=[fc_hidden_channels],
            num_output_channels=num_output_channels,
            activation=activation
        )

    def forward(self, tuple_inputs):
        latents, image = tuple_inputs
        image = self.activation(self.resnet(image))

        linear_input = torch.cat((latents, image.flatten(start_dim=1)), dim=1)
        output = self.mlp(linear_input)
        return output


class MaskedLinear(nn.Module):
    def __init__(self, input_degrees, output_degrees):
        super().__init__()

        assert len(input_degrees.shape) == len(output_degrees.shape) == 1

        num_input_channels = input_degrees.shape[0]
        num_output_channels = output_degrees.shape[0]

        self.linear = nn.Linear(num_input_channels, num_output_channels)

        mask = output_degrees.view(-1, 1) >= input_degrees
        self.register_buffer("mask", mask.to(self.linear.weight.dtype))

    def forward(self, inputs):
        return F.linear(inputs, self.mask*self.linear.weight, self.linear.bias)


class AutoregressiveMLP(nn.Module):
    def __init__(
            self,
            num_input_channels,
            hidden_channels,
            num_output_heads,
            activation
    ):
        super().__init__()
        self.flat_ar_mlp = self._get_flat_ar_mlp(num_input_channels, hidden_channels, num_output_heads, activation)
        self.num_input_channels = num_input_channels
        self.num_output_heads = num_output_heads

    def _get_flat_ar_mlp(
            self,
            num_input_channels,
            hidden_channels,
            num_output_heads,
            activation
    ):
        assert num_input_channels >= 2
        assert all([num_input_channels <= d for d in hidden_channels]), "Random initialisation not yet implemented"

        prev_degrees = torch.arange(1, num_input_channels + 1, dtype=torch.int64)
        layers = []

        for hidden_channels in hidden_channels:
            degrees = torch.arange(hidden_channels, dtype=torch.int64) % (num_input_channels - 1) + 1

            layers.append(MaskedLinear(prev_degrees, degrees))
            layers.append(activation())

            prev_degrees = degrees

        degrees = torch.arange(num_input_channels, dtype=torch.int64).repeat(num_output_heads)
        layers.append(MaskedLinear(prev_degrees, degrees))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        assert inputs.shape[1:] == (self.num_input_channels,)
        result = self.flat_ar_mlp(inputs)
        result = result.view(inputs.shape[0], self.num_output_heads, self.num_input_channels)
        return result


