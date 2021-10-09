'''
Code to support understanding in Depthwise separable convolution.
'''

import torch.nn as nn


def calculate_parameters(operation):
    return sum(param.numel() for param in operation.parameters() if param.requires_grad)


in_channels = 8  # M (Depth of filter)
out_channels = 128  # N (number of filters)
Dk = 3  # Kernel Size

# Standard Convolution
standard_conv = nn.Conv2d(in_channels, out_channels, kernel_size=Dk, padding=1, bias=False)
sc_total_param = calculate_parameters(standard_conv)

# Depthwise Separable Convolution
depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=Dk, padding=1, bias=False,
                           groups=in_channels)  # To be clear, depthwise convolution is done by groups=in_channels, refer to first note: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

depthwise_separable_convolution = nn.Sequential(
    depthwise_conv,
    pointwise_conv
)
ds_total_param = calculate_parameters(depthwise_separable_convolution)

print('\n')
print(f'Total learnable parameters in standard convolution: {sc_total_param}')
print(f'Total learnable parameters in depthwise separable convolution: {ds_total_param}')

# Let's check the maths, 1/N + 1/Dk**2 should be the reduction value

computation_reduction = ds_total_param / sc_total_param
equation_reduction = 1 / out_channels + 1 / (Dk ** 2)

print('\n')
print(f'Equation (1/N + 1/Dk**2) = {equation_reduction}')
print(f'PyTorch Computation Reduction calculation = {computation_reduction}')

# 1/reduction = multiplier (paper says between 8 to 9 cause it is dependent on the number of filters we are using)
print(
    f'Standard convolution has {sc_total_param / ds_total_param} higher computational cost than depthwise separable convolutions')
