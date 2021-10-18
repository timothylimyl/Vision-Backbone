# Vision Backbones

[Under constant editing, 14/10/21]

Vision backbones networks serves as the feature extractor of input image data. It encodes
features into a latent space representation which can be utilised/decoded by many downstream vision tasks like 
image classification, segmentation, object detection, pose estimation, and many more. In other words, vision backbones simplifies the representation of the input data
into a compressed representation that is linked to the values and structure of the input data. The compressed feature representation has to be robust to a 
multitude of changes in the input data. For example, the encoded feature space for images of different breed of dogs will be close to each other despite the 
pixel values of each image being very different from each other. The vision backbone's goal is to build a compressed feature representation that generalised to a 
wide range of pixel values to allow for downstream analysis such as trying to classify dogs from other objects. The quality of the encoded feature representation from the backbones directly affects the performance of any downstream tasks. Thus, it is imperative research and engineering
work to improve the performance of vision backbones.

## Purpose

The focus of this repository to **explore interesting and impactful deep learning based vision backbone ideas.** 
The main goal will be to pick up ideas from research papers and implement them in code to
fully comprehend the proposed ideas. I believe that by reading and implementing 
the vision backbone networks proposed by the research community, we too can formulate our own ideas and 
do better engineering work instead of treating the outcome of research papers as plug and play.

My core focus will be figuring the inspiration of the paper, the core contribution made and the implementation in code.


## Vision Backbones through the years


### LeNet5, Year 1989 

[GradientBased Learning Applied to Document
Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)


Pioneering paper that shows the potential applicability of Convolutional Neural Network (CNN) as a vision feature extractor 
(backbone). For this paper, the particular task was identifying handwritten zip code numbers. 
At this time, GPU and big datasets for training are non-existent which makes it hard for researcher to show any substantial results using 
CNN as neural networks are highly parameterised. Neural network requiring a lot of computational power and a lot of data for it to learn.
However, neural networks structure are embrassingly parallel. Thus, a dedicated hardware such as GPUs that optimises for throughput instead of latency will be
greatly beneficial to training and testing neural networks.

[Implementation notes, lenet5.py](): The Hello World of CNN (1)


### AlexNet, Year 2012 

[ImageNet Classification with Deep Convolutional
Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

Paper showing that CNN can achieve astounding results (won [ILSVRC](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf)) and optimised GPU implementations of neural networks 
can achieve super fast training speed. It must be noted that the delayed rise of CNN (post-2012) can be attributed to GPU capability and the lack of development in this space.
As we see today (2021), GPUs has tremendous compute capability and open-source Deep Learning frameworks 
has provided many core GPU based implementation (cuda ops) to build various kinds of neural networks.

Another notable contributions are ReLu activation functions to accelerate the 
training process and also the usage of Dropout to combat overfitting. 
The creation of ImageNet dataset (~2009) with an abundance of images helped a lot as CNN relies on it to 
outperform traditional methods by a large margin. Hence, started the shift from manual feature engineering to network engineering.

[Implementation notes, alexnet.py :]() The Hello World of CNN (2)

### VGG, Year 2014 

[Very Deep Convolutional Networks For Large-scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)

Paper contributed that small convolution filters evidently give better results. 
AlexNet first layer is 7x7 filter with stride 4  which on hindsight is a bad idea especially since the
first layer literally extracting out features directly from the input images where you want the finest of
features be extracted so that network can build better feature representations in later layers. VGG went
with 3x3 filters with stride of 1 (small receptive fields) which is pretty much the common standard filte
r size today. 3x3 filters also has the benefit of having less trainable parameters than a 7x7 filters under equal effective 
receptive field (3 stacks of 3x3 vs 1 stack of 7x7) and my general intuition is that 3x3 filters allows more distinct kernel out
comes. Results of the paper also reaffirms that deeper networks can outperform shallower networks (depth is important). 

[Implementation notes, vgg16.py :]() The Hello World of CNN (3)

[Implementation notes, vgg16_clean.py :]() The Hello World of CNN (4). From here on, it can be observed that a lot of CNN design has
repetitive layers and the cleaner way to construct the code will be to create a general class/function whenever you feel
that you are copying and pasting a lot of code.


### Inception, Year 2014 

[Going deeper with convolutions
](https://arxiv.org/pdf/1409.4842.pdf)

Focus of the author is to contribute a design of network that is optimised for both performance and computational efficiency. 
First of, to improve performance, Inception model uses a series of kernels with different sizes to handle multiple feature scales. Secondly, for computational efficiency, 1x1 convolutions are introduced to reduce dimensionality. The paper uses 1x1 convolution to compute reductions before 3x3 and 5x5 convolutions which are computationally expensive operations. In simple terms, 1x1 reduces the amount of channels so that 3x3 and 5x5 convolutions convolves on feature maps with less channels. The core idea of Inception is going "wide", having multiple layers convolving the same feature map with different kernel sizes and concatenate the outputs together. Personally, I think it's intuitive to think that this provides the model with more choices when learning to formulate different feature representations.
In the following year (2015), the improvements of [Inception v2-v3](https://arxiv.org/pdf/1512.00567.pdf) still focuses on the computational efficiency of the network while also scaling performance. I like that the authors is not scaling the networks naively. The authors proposed to factorised into smaller convolutions by changing all 5x5 filters to two 3x3 filter instead. This idea is already present in the VGG paper, 3x3 filters becomes pretty much a standard kernel size for any network.

A brilliantly simple and neat idea of inception is the use of auxiliary classifier which connects to intermediate layers for
the contribution of the total loss. This is assuming that the intermediate layers connected to the auxiliary classifier is 
sufficiently discriminative to make meaningful predictions. The difference of the auxiliary classifier and the network in 
Inception-v1 and Inception-v3 is that v3 uses batch normalisation in its layers (next up: batch norm!).


A warning that we all should heed (from the paper) as we read Deep Learning papers:

> One must be cautious though: although the proposed architecture has become a success for computer vision, 
> it is still questionable whether its quality can be attributed to the guiding principles that have lead to its construction.


[Implementation notes, inception.py :]() I followed Inception v1 paper to build the code, Inception v2-v3 paper lack details in terms of the amount of channels
for each layer so it will be hard for users to read the paper and follow the code. However, the general idea is the same, 
learn how to code for multiple branches and then how you want operate on each branch can be up to you as long as 
you ensure the feature map size is the same so that you can concatenate the branches together.

### [Technique] Batch Normalisation, Year 2015 


[Batch Normalization: Accelerating Deep Network Training b
y
Reducing Internal Covariate Shift
](https://arxiv.org/pdf/1502.03167.pdf)

The input distribution of each layer changes during every training update which slows down training according to the authors of the paper. 
They proposed to normalised the input distribution that is given to every layer as the network propagates forward. 
It was found that by normalising the input distribution, we can use higher training rates (train faster to higher accuracy) 
and also provides regularisation (prevent overfitting). It is worth noting that normalising the data inputs of first layer of the network to 
improve training speed has its roots way back to [a paper in 1998, Section 4](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf).
Introduction of batch normalisation is to apply the same concept to every layer.

Batch normalisation (BN) is done over a mini-batch (stating the obvious). 
We subtract the layer's input values by the mean of the batch and divide by standard deviati
on of batch. Finally, we allow the normalised values to scale and shift by multiplying it by a 
trainable parameter (scale) and adding another trainable parameter (shift). 
Thus, it is important to note that the original input distribution can actually be completely restored to the same distribution if the batch normalisation trainable parameters is updated to the batch mean and standard deviation value (scale and shift back). 

Personally, the main takeaways from the paper is that BN just works great so use it everywhere! 
The experiments carried out in the paper prove that applying BN can not only achieve higher accuracy 
but allows faster training. Go back to VGG and apply BN `nn.BatchNorm2d()` after every activation 
function and you will most probably see that your model has improved, this is proven by experiments
done [here](https://pytorch.org/vision/stable/models.html) if you do not have the resources to try it out. This paper has proven it using the Inception model.
 
[How does it help?](https://arxiv.org/abs/1805.11604)


### ResNet, Year 2015 

[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

It was proven that adding more layers up to a certain point actually could lead to an even higher 
training error. Theoretically, the more parameters you have, the better the model can approximate 
([Universal Approximation Theorem]()). 
However, our optimisation routine is not perfect, deeper networks are harder to optimised to the point
of global minimum. ResNet tackles this optimisation problem (problem of vanishing gradients) that 
hampers optimisation convergence during training by using Residual Networks. 
Residual Networks are basically "shortcut" connections that skips layers as network propagates forward.
My intuition is that residual networks are an additional pathway for information (gradient values) to 
flow;  allowing the model with some flexibility during optimisation to bypass certain layers.
The great thing about residual connections is that it does not add any extra parameters, comparison to 
plain CNN can be made (ex: ResNet vs VGG with an exact set up). With usage of residual networks, extra layers can be added to improve model performance; once again, the same caveat is that improvement is only up to a certain network depth.


[Implementation notes, resnet.py :]() The code only consist of set up for ResNet18 and ResNet34, ResNet with more layers uses a different type of block which I
am currently not interested in as it is pretty much similar to the basic block idea with extra layers and the application of 1x1 convolution
for bottlenecking (reduction in depth, same idea as proposed in Inception).

###  [Technique] Spatial Pyramid Pooling, Year 2015

[Spatial Pyramid Pooling in Deep Convolutional
Networks for Visual Recognition](https://arxiv.org/pdf/1406.4729.pdf)

With the introduction of Spatial Pyramid Pooling (SPP) to CNN, we no longer need to rely on a fixed image size input. SPP is used
at the last convolutional layer to pool together feature maps in a standardised method to ensure a fixed output representation irregardless
of input size. This is done by using multiple **fixed** pooling filter kernels of different sizes to pool the feature maps
into a standard output size. As the output of SPP is of a standard size, the amount of fully connected layer nodes that is connected
to the output of the SPP is fixed and invariant to the size of the input image. As we are not constraint by input image size, we can flexibly
change the image input size during training to provide different image scales to learn from. 


[Implementation notes, spp.py :]() SPP only requires 2 information, the number input channels to the SPP layer and the different pooling scales that you want to use. 
In the code, you can adjust the height and width of the input, SPP will ensure that a fixed size representation is given to the
fully connected layers. The max pooling operations used to encode the features into a fixed size representation does
not require us to strictly follow it, we can flexibly change the all of the operations
used in SPP.


---
### Pause and reflect

After reading up to here (End of Year 2015), we can easily deduced that the low hanging fruit for researchers 
seeking to improve the models and publish papers in this domain is to combine the idea of efficient 
multi-branch kernels (going wide like Inception) and applying it to networks like ResNet (vice versa). 
Researchers can design different network width and depth configuration and check for improvement in the 
model performance. 

I won't go into these networks at least for now. You can follow this train of ideas/papers starting from:

- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf)
- [Inception-ResNet](https://arxiv.org/pdf/1602.07261.pdf)

Also, following from ResNet novel idea of amending the connections from layer to layer, researchers can formulate
more unique connections design to allow better information flow (easier for the model to optimise).

---


### DenseNet, Year 2016 

[Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)

Instead of combining features through summation like in ResNet, DenseNet proposes to concatenate the
features. However, to concatenate the feature maps, you can only do it if the feature map size is the 
same. The paper seperate the network into Dense Block and Transition Layer; Dense Block are basically 
a stack of 1x1 and 3x3 convolution operations while Transition Layer is the downsampling layer 
(typical convolution and pooling to get smaller feature maps). 
Main thing to note in the Dense block is that the feature maps of a layer connects to every layer in 
front of it. Said in another way, each feature map in the Dense Block has access to all preceding 
feature maps. This allows for feature reusing during learning. 
Thus, the usage of learnable parameters is more efficient due to this dense connection set up 
(in other words, less redundant parameters in the model). 

[Implementation notes, densenet.py :]() Implemented following the paper, the paper is thankfully very clear on its
implementation and the same configuration scales to all unlike ResNet that introduces a new block. To ensure correctness, 
I debugged layer/block by layer/block to ensure that the feature map size matches the paper.

### [Technique] SENet/Block 2017 


[Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)

Most vision backbones networks to date (pre-2017) does not provide a learning pathway for the network 
to internally learn how to weight/prioritise features that are deemed more important to make the 
correct prediction. It is obvious that not every feature map will be important/necessary for every 
image prediction; for example, if we are trying to classify a dog, background features does not matter
much to us whereas if we are classify location then background features are of importance. 
Thus, it can be useful if the network itself learns how to adaptively prioritise its learnt feature 
representation which enchances representational power of the network; this is commonly known as Soft Attention. 

SENet introduces a form of Soft Attention that is computationally cheap and can be easily adapted to 
any vision backbones. The idea is to provide an extra pathway (SE Block) of learnable parameters that
focuses on learning channels weights. The channels weights is used to calibrate channel-wise features 
response by simply multiplying its weights to the feature maps. Thus, we need to ensure that the output
of a SE Block will be of the same depth/channels as the feature maps to weight every channel.
It is very intuitive that giving the model the ability to weight its features during learning will 
be beneficial as it can learn to prioritise (greater weight means a higher relevancy). 
In terms of SE Block, it is learning to prioritise channel-wise relationship 
(channel attention mechanism). 
It was found that at earlier layers, SEBlocks will be selective of informative features 
that are class-agnostic as early feature maps are more generalised and shared by different classes. 
However, as we go to deeper layers, the SEBlocks will be selective of class-specific representations. 
The allowance of this learnt selectivity strengthens the discriminative performance of the model. 
Take note that at the last stages of the network, the author found that the SEBlock reduces down to 
just a mere scaling or identity function which does not contribute much to recalibrating the feature maps. Thus, we can experiment ourselves to remove SEBlocks from our last few layers to verify whether is performance degradation significant to the reduction of parameters. (Highly recommended to check out the graphs in the paper (Figure 6))

I am putting SEBlock as more of a technique.
The paper has proven that added SEBlock to vision backbones such as ResNet, ResNeXt and Inception 
will boost its performance and the increase in GFLOPs is not significant (minimal overhead!). 

[Implementation notes, SEblock.py :]() I set up a simple standard convolution operation block and then added a SEBlock to it. You can run the code
and observe that the parameters added by the SEBlock is not much. The code separates Squeeze and Excite operation clearly to aid understanding.


### [Technique] Depthwise seperable convolutions, MobileNet, 2017 

[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications](https://arxiv.org/pdf/1704.04861.pdf)

The main goal of MobileNet is to formulate efficient CNN for the lightweight application such as mobile phones and embedded devices. My main interest in MobileNet is the introduction of factorising standard convolution in CNN to depthwise convolution and pointwise convolution. It has proven to be an efficient tradeoff of model performance to model and computational size. The other contribution is add 2 simple parameter to adjust the depth of feature maps and input resolution according to the needs of the user which is basic engineering work.

Standard convolution multiplies/filters and add inputs together in a single step. Depthwise seperable convolutions seperates the filtering step (depthwise convolution) and the combining step (pointwise convolution). Depthwise convolutions applies a single filter per feature map channel while pointwise convolution combines/adds the output of the depthwise convolutions. MobileNet applies batch normalisation and relu activation functions for respective convolution operation.

Main important finding from the paper:

> MobileNet uses 3 × 3 depthwise separable convolutions which uses between 8 to 9 times less
> computation than standard convolutions at only a small reduction in accuracy


The mathematics for computational cost works out very easily. 

* Assumption: Convolutions uses stride of 1 and same padding.
* Given: Input feature map (`Df`), Channels (`M`), Number of convolution filters (`N`), Kernel size (`Dk`)

Standard convolution computational cost: `Dk x Dk x M x N x Df x Df`

Depthwise convolution -  `Dk x Dk x M x Df x Df`
Pointwise convolution -    `M x N x Df x Df`
Depthwise separable convolutions:  `Dk x Dk x M x Df x Df   +  M x N x Df x Df`

Factorising the reduction in computation by using ratio Depthwise seperable over Standard convolution, 
we will get 

`(Dk x Dk + N)/ (N x DK x DK)  = 1/N + 1/(DK x DK)`

I personally think of it this way to clearly understand the reduction in parameters: 
Every pixel point on EVERY feature maps will go through `Dk x Dk x M x N` cost for
standard convolution, while every pixel point on EVERY feature maps will only go through `Dk x Dk x M`
for depthwise convolution. Pointwise convolution then only cost `N x 1 x 1` for every pixel point 
on every feature maps. This makes it clear that standard convolution computation cost rises relative 
to depthwise separable by Channels of feature maps (`M`) and number of convolution filters (`N`).

The findings from MobileNet is not just for when we are computationally constrained 
by hardware. We should just depthwise separable convolutions for any application as it has proven to 
be more efficient. One of the experiment that I think MobileNet should had ran was adding more layers 
to match standard convolution network parameters and operations to compare accuracy. 
For example, they can convert the residual blocks of ResNet-18 to depthwise separable convolutions
and then add more layers till it matches the original ResNet-18 parameters and operations. 
They only ran experiments showing the difference between a standard convolution mobilenet versus
Mobilenet (Table 4 of paper) With that being said, their experiments has already proven that at way 
lesser parameters and operations, the performance of MobileNet is only slightly reduced.


[Implementation notes, depthwise_separable.py :]() Code to aid understanding in the difference between standard convolution and depthwise separable convolution. 
Let the code do the maths for you, play around with the number of filters and channels to gain some intuition.

## Future works

Nothing determined. I will write and code papers that I find interesting, novel and useful.

[18/10: Of current interest]

1. HRNet
2. CBAM
3. ResNest
4. ViT
5. Swin















