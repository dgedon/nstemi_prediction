import os
import torch
import torch.nn as nn
import functools
import operator
import copy
import numpy as np
from tqdm import tqdm


# -----------------------------------------------------------------------------
# ResNet Model
# -----------------------------------------------------------------------------
class SELayer(nn.Module):
    """adapted from https://github.com/moskomule/senet.pytorch"""

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # self.bn = nn.BatchNorm1d(channel)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        # y = self.bn(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError(
            "Number of samples for two consecutive blocks "
            "should always decrease by an integer factor."
        )
    return downsample


class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(
        self,
        n_filters_in,
        n_filters_out,
        downsample,
        kernel_size,
        dropout_rate,
        is_first_block=False,
        dont_use_se_layer=False,
        se_reduction=16,
    ):
        if kernel_size % 2 == 0:
            raise ValueError(
                "The current implementation only support odd values for `kernel_size`."
            )
        super(ResBlock1d, self).__init__()
        self.is_first_block = is_first_block
        self.use_se_layer = not dont_use_se_layer
        self.relu = nn.ReLU()

        # conv-block 1
        if not self.is_first_block:
            self.bn1 = nn.BatchNorm1d(n_filters_in)
            self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(
            n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False
        )

        # conv-block 2
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(
            n_filters_out,
            n_filters_out,
            kernel_size,
            stride=downsample,
            padding=padding,
        )  # , bias=False)

        # SE layer
        if self.use_se_layer:
            self.se_layer = SELayer(n_filters_out, se_reduction)  # n_filters_in

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1)  # , bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x):
        """Residual unit."""
        if self.skip_connection is not None:
            x_skip = self.skip_connection(x)
        else:
            x_skip = x
        # 1st layer
        if not self.is_first_block:
            # do not add these for the first residual block
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout1(x)
        x = self.conv1(x)
        # 2nd layer
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.conv2(x)

        # SE layer
        if self.use_se_layer:
            x = self.se_layer(x)
        # sum main path with skip connection
        x += x_skip
        return x


class ResNet1d(nn.Module):
    """Residual network for unidimensional signals.

    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.5
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.

    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    """def __init__(self, input_dim, blocks_dim, kernel_size=17, dropout_rate=0.5,"""

    def __init__(
        self,
        input_dim,
        filter_size,
        downsampling,
        kernel_size=17,
        dropout_rate=0.5,
        dont_use_se_layer=False,
        se_reduction=16,
    ):
        super(ResNet1d, self).__init__()
        self.relu = nn.ReLU()

        # First layer
        n_filters_in, n_filters_out = input_dim[0], filter_size[0]
        padding = _padding(downsampling[0], kernel_size)
        self.conv1 = nn.Conv1d(
            n_filters_in,
            n_filters_out,
            kernel_size,
            bias=False,
            stride=downsampling[0],
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = []
        for i, (fs, ds) in enumerate(zip(filter_size[1:], downsampling[1:])):
            n_filters_in, n_filters_out = n_filters_out, fs
            # first block is different
            first_block = True if i == 0 else False
            resblk1d = ResBlock1d(
                n_filters_in,
                n_filters_out,
                ds,
                kernel_size,
                dropout_rate,
                is_first_block=first_block,
                dont_use_se_layer=dont_use_se_layer,
                se_reduction=se_reduction,
            )
            self.add_module("resblock1d_{0}".format(i), resblk1d)
            self.res_blocks += [resblk1d]

        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Residual blocks
        for blk in self.res_blocks:
            x = blk(x)

        # final BN, ReLU, dropout
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x


# -----------------------------------------------------------------------------
# Encoding of age and sex
# -----------------------------------------------------------------------------
class AgeSexEncoding(nn.Module):
    def __init__(self, output_dim=128):
        super(AgeSexEncoding, self).__init__()
        self.output_dim = output_dim

        # linear layer
        self.linear = nn.Linear(2, self.output_dim)
        #
        self.relu = nn.ReLU()

    def forward(self, age_sex):
        # linear layer
        out = self.linear(age_sex)
        out = self.relu(out)

        return out


# -----------------------------------------------------------------------------
# Linear Prediction Stage
# -----------------------------------------------------------------------------
class LinearPredictionStage(nn.Module):
    def __init__(self, model_output_dim, n_classes):
        super(LinearPredictionStage, self).__init__()

        self.lin_classifier = nn.Linear(model_output_dim, n_classes)

    def forward(self, x):
        # Fully connected layer
        x = self.lin_classifier(x)

        return x


# -----------------------------------------------------------------------------
# ECG Model (+helper functions)
# -----------------------------------------------------------------------------
def get_resnet(config):
    num_res_blks_options = {
        "blk_sizes": [4, 8, 12],
        "net_filter_size": [
            [64, 128, 196, 256, 320],
            [64, 128, 128, 196, 256, 256, 320, 512, 512],
            [64, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512],
        ],
        "net_downsample_factors": [
            [1, 4, 4, 4, 4],
            [1, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2],
        ],
    }

    n_input_channels = config.num_leads
    # get net filter size and downsample factor from the appropriate number of blocks
    if config.num_resnet_blks != -1:
        idx = num_res_blks_options["blk_sizes"].index(config.num_resnet_blks)
        net_filter_size = num_res_blks_options["net_filter_size"][idx]
        net_downsample_factors = num_res_blks_options["net_downsample_factors"][idx]
    else:
        net_filter_size = config.net_filter_size
        net_downsample_factors = config.net_downsample_factors
    # check if input sequence length and downsampling is compatible
    final_length = config.seq_length // functools.reduce(
        operator.mul, net_downsample_factors
    )
    if not final_length > 1:
        ValueError("Input sequence length not compatible with downsampling factors.")

    blocks_downsampling = [net_downsample_factors[0]]
    blocks_filter_size = [net_filter_size[0]]
    for fs, ds in zip(net_filter_size[1:], net_downsample_factors[1:]):
        blocks_filter_size.extend([fs])
        blocks_downsampling.extend([ds])

    # Get resnet backbone model
    resnet = ResNet1d(
        input_dim=(n_input_channels, config.seq_length),
        filter_size=blocks_filter_size,
        downsampling=blocks_downsampling,
        kernel_size=config.kernel_size,
        dropout_rate=config.dropout_resnet,
        dont_use_se_layer=config.dont_use_se_layer,
        se_reduction=config.se_reduction,
    )

    return resnet, net_filter_size[-1], final_length


class ECGModel(nn.Module):
    def __init__(self, config):
        super(ECGModel, self).__init__()
        self.device = config.device
        
        # get resnet model parts
        self.resnet, final_filter_size, final_length = get_resnet(config)

        # get age and sex embeddings
        self.age_sex_emb = AgeSexEncoding(config.age_sex_output_dim)
        combined_output_dim = (
            final_filter_size * final_length + config.age_sex_output_dim
        )

        # get final prediction stage
        self.pred_stage = LinearPredictionStage(
            model_output_dim=combined_output_dim,
            n_classes=config.num_outputs,
        )

    def forward(self, inp):
        # unpack
        traces, age_sex = inp

        # resnet forward
        features = self.resnet(traces)
        # Flatten array
        features = features.view(features.size(0), -1)

        # embeddings forward
        emb_out = self.age_sex_emb(age_sex)

        # combine embeddings with resnet output
        features = torch.cat([emb_out, features], dim=1)
        # prediction stage
        logits = self.pred_stage(features)
        return logits, features


# -----------------------------------------------------------------------------
# ECG Ensemble Model
# -----------------------------------------------------------------------------
class EnsembleECGModel(ECGModel):
    def __init__(self, config, log_dir):
        super(EnsembleECGModel, self).__init__(config)

        self.trained_model_dir = log_dir
        self.model_list = self.load_model_list(config)

    def load_model_list(self, args):
        # load the best models for each ensemble member
        model_list = []
        for i in range(1, args.num_ensembles + 1):
            # load generic model
            model = ECGModel(args)
            # put to evaluation model
            model.eval()
            # load stored weights
            model_path = os.path.join(self.trained_model_dir, f"model_{i}.pth")
            map_location = {
                "cuda:%d"
                % 0: (
                    "cuda:%d" % self.device.index
                    if self.device.index is not None
                    else "cuda"
                )
            } if torch.cuda.is_available() else torch.device("cpu")
            state_dict = self.convert_ddp_model_parameters(
                torch.load(model_path, map_location=map_location, weights_only=True)["model"]
            )
            model.load_state_dict(state_dict)
            model_list.append(copy.deepcopy(model))
        return model_list

    @staticmethod
    def convert_ddp_model_parameters(state_dict):
        """
        converts the parameters of a model saved with DDP to a model without DDP framework.
        basically removes "module." from the start of the parameter name since DDP stored models start with "module."
        """
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.'
            new_state_dict[name] = v
        return new_state_dict

    def forward(self, inp):
        # allocation
        logits_list = []
        features_list = []

        pbar = tqdm(self.model_list, total=len(self.model_list), desc="Ensemble model", leave=False)
        # model forward for each ensemble member
        for model in pbar:
            # set model
            self.set_model_member(model)
            # run forward pass
            logits, features = model.forward(inp)
            # append
            logits_list.append(logits)
            features_list.append(features)

        # average logits
        logits = torch.stack(logits_list).mean(dim=0)
        features = torch.cat(features_list, dim=-1)

        # output logits
        return logits

    def set_model_member(self, model):
        self.resnet = model.resnet.to(device=self.device)
        self.age_sex_emb = model.age_sex_emb.to(device=self.device)
        self.pred_stage = model.pred_stage.to(device=self.device)
