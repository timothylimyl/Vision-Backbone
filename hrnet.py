import torch
from torch import nn
import torch.nn.functional as F
import torch.onnx

# GLOBALS:
BN_MOMENTUM = 0.1


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c):
        super(StageModule, self).__init__()

        self.number_of_branches = stage  # number of branches is equivalent to the stage configuration.
        self.output_branches = output_branches

        self.branches = nn.ModuleList()

        # Note: Resolution + Number of channels maintains the same throughout respective branch.
        for i in range(self.number_of_branches):  # Stage scales with the number of branches. Ex: Stage 2 -> 2 branch
            channels = c * (2 ** i)  # Scale channels by 2x for branch with lower resolution,
            # 2x multiplier scales moving from one branch to the next branch.

            # Paper does x4 basic block for each forward sequence in each branch (x4 basic block considered as a block)
            branch = nn.Sequential(*[BasicBlock(channels, channels) for _ in range(4)])

            self.branches.append(branch)  # list containing all forward sequence of individual branches.

        # For each branch requires repeated fusion with all other branches after passing through x4 basic blocks.
        self.fuse_layers = nn.ModuleList()

        for branch_output_number in range(self.output_branches):
            # Number of outputs u want from branch, typically equal to number of branches
            # HRNet author decided to only output the high resolution stream in final layer.
            # so for the final layer, the number of output from branch is 1 despite having
            # numerous branches.
            # Ex: for final layer, you only want output from 1 branch so you only need to get the fusion layers for
            #     that particular branch.
            # note: branch_output_number indicates the particular branch we are focusing on to get the fusion layers off
            #       all other branches (branch_number) to be outputted.

            self.fuse_layers.append(  # each fuse layer takes care of one output stream/branch.
                nn.ModuleList())  # fuse layers keeps track of all streams fusion inputs to fuse with the current stream to output

            for branch_number in range(self.number_of_branches):
                # Number of branches, need go through each to get the fusion layers for each respective stream/branch output (branch output_number)
                # branch_number indicates the particular branch u are looking at for fusion to the branch output (branch_output number)

                # As you are looking at the branch_output_number which is the resolution stream, you want to fuse other branches
                # to it. So there is 3 logic:
                #  1. If you are on the same branch, no fusion will occur
                #  2. If you are looking at a lower resolution stream (branch_number > branch_output_number), you want to
                #     upsample the lower stream to fuse with your current branch
                #  3. If you are looking at a higher resolution stream (branch_number < branch_output_number), you want to
                #     downsample the higher stream to fuse with your current branch.

                if branch_number == branch_output_number:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif branch_number > branch_output_number:
                    # upsampling logic: you will need to match the channels of the higher resolution stream/branch.
                    # the reason why we scale by 2 to the power of the branch_output_number and branch_number
                    # is because we scale the channels by 2 to the power off the branch gaps in the transition layer,
                    # so we just recovering the same way as how we do the transition layers to get lower resolution streams.
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2 ** branch_number), c * (2 ** branch_output_number), kernel_size=1, stride=1,
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** branch_output_number), eps=1e-05, momentum=0.1, affine=True,
                                       track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (branch_number - branch_output_number)), mode='nearest'),
                    ))
                elif branch_number < branch_output_number:
                    # downsampling logic: you will need to match the channels of the lower resolution stream/branch.

                    # We want to repeat 3x3 conv, stride 2, channels=number of channels for the branch. Repeat according to
                    # the number of transition layers used for that particular high resolution stream minus 1 and
                    # then only downsample using 1x1 conv to the number of channels of the lower resolution stream.
                    # example: if we are at branch_output_number 3, to get fusion layer from the highest resolution branch
                    #          (branch_number=0), we will repeat 3x3 conv,stride2, channels=c for 2 times and then use 1x1
                    #          conv to match the number of channels in the lower resolution stream (branch_output_number)

                    downsampling_fusion = []
                    for _ in range(branch_output_number - branch_number - 1):
                        downsampling_fusion.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** branch_number), c * (2 ** branch_number), kernel_size=3, stride=2,
                                      padding=1,
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** branch_number), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    downsampling_fusion.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** branch_number), c * (2 ** branch_output_number), kernel_size=3,
                                  stride=2, padding=1,
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** branch_output_number), eps=1e-05, momentum=0.1, affine=True,
                                       track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*downsampling_fusion))

                    # note: fuse_layers is a list indexing all fusion layers for each streams.
                    # ex: self.fuse_layers[0] will be the fusion layers used for fusing to the highest resolution stream.
                    # fuse_layers[0][0] will be the specific fusion layer to fuse a specific stream to the highest resolution stream.
                    # which in this case is no fusion cause u are already at the highest resolution stream.

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        assert len(self.branches) == len(x)
        # input to each stage is a list of inputs for each branch
        x = [branch(branch_input) for branch, branch_input in zip(self.branches, x)]

        x_fused = []
        for branch_output_index in range(
                self.output_branches):  # Amount of output branches == total length of fusion layers
            for input_index in range(self.number_of_branches):  # The inputs of other branches to be fused.
                if input_index == 0:  # Going to each new branch output (new branch_output_index), append to the list first.
                    x_fused.append(self.fuse_layers[branch_output_index][input_index](x[input_index]))
                else:
                    x_fused[branch_output_index] = x_fused[branch_output_index] + self.fuse_layers[branch_output_index][
                        input_index](x[input_index])

        # After fusing all streams together, you will need to pass the fused layers
        for i in range(self.output_branches):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused  # returning a list of fused outputs


class HRNet(nn.Module):
    def __init__(self, c=48):
        super(HRNet, self).__init__()

        # Stem:
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=BN_MOMENTUM, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=BN_MOMENTUM, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1:
        # You need to downsample first because you are connecting to bottleneck module (reference >resnet50)
        # Thus, for residual connections to be made for the first layer, you need to match the channels size.
        # This is exactly what you will do for the first bottleneck layer in resnet or whenever u want to downsample
        # and increase channels.
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=BN_MOMENTUM, affine=True, track_running_stats=True),
        )
        # Note that bottleneck module will expand the output channels according to the output channels*block.expansion
        bn_expansion = Bottleneck.expansion # The channel expansion is set in the bottleneck class.
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),  # Input is 64 for first module connection
            Bottleneck(bn_expansion * 64, 64),
            Bottleneck(bn_expansion * 64, 64),
            Bottleneck(bn_expansion * 64, 64),
        )

        # Transition 1 - Creation of the first two branches (one full and one half resolution)
        # Need to transition into high resolution stream and mid resolution stream
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c, eps=1e-05, momentum=BN_MOMENTUM, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(256, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c * 2, eps=1e-05, momentum=BN_MOMENTUM, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)  - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        number_blocks_stage2 = 1
        self.stage2 = nn.Sequential(
            *[StageModule(stage=2, output_branches=2, c=c) for _ in range(number_blocks_stage2)])

        # Transition 2  - Creation of the third branch (1/4 resolution)
        self.transition2 = self._make_transition_layers(c, transition_number=2)

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        number_blocks_stage3 = 4  # number blocks you want to create before fusion
        self.stage3 = nn.Sequential(
            *[StageModule(stage=3, output_branches=3, c=c) for _ in range(number_blocks_stage3)])

        # Transition  - Creation of the fourth branch (1/8 resolution)
        self.transition3 = self._make_transition_layers(c, transition_number=3)

        number_blocks_stage4 = 2  # number blocks you want to create before fusion
        self.stage4 = nn.Sequential(
            *[StageModule(stage=4, output_branches=4, c=c) for _ in range(number_blocks_stage4)])

    def _make_transition_layers(self, c, transition_number):
        return nn.Sequential(
            nn.Conv2d(c * (2 ** (transition_number - 1)), c * (2 ** transition_number), kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(c * (2 ** transition_number), eps=1e-05, momentum=BN_MOMENTUM, affine=True,
                           track_running_stats=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Stem:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Stage 1
        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # split to 2 branches, form a list.

        # Stage 2
        x = self.stage2(x)
        x.append(self.transition2(x[-1]))

        # Stage 3
        x = self.stage3(x)
        x.append(self.transition3(x[-1]))

        # Stage 4
        x = self.stage4(x)

        # HRNetV2 Example: (follow paper, upsample via bilinear interpolation and to highest resolution size)
        output_h, output_w = x[0].size(2), x[0].size(3)  # Upsample to size of highest resolution stream
        x1 = F.interpolate(x[1], size=(output_h, output_w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x[2], size=(output_h, output_w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x[3], size=(output_h, output_w), mode='bilinear', align_corners=False)

        # Upsampling all the other resolution streams and then concatenate all (rather than adding/fusing like HRNetV1)
        x = torch.cat([x[0], x1, x2, x3], dim=1)

        return x


if __name__ == '__main__':
    model = HRNet(48)
    # model = HRNet(32)

    if torch.cuda.is_available() and False:
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model = model.to(device)
    y = model(torch.ones(1, 3, 384, 288).to(device))
    print(y.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
