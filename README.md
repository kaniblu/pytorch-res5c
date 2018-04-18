# Extracting ResNet Features in PyTorch #

In many vision-related tasks, extracting and utilizing pre-trained visual 
features from  images in question is a common practice. However, doing so is not 
so as straight-forward as one could hope for in pytorch, as non-linear 
activation layers are embedded in building blocks. This simple script should 
reduce some unwarrented overhead for others who are also trying to achieve the 
same goal.

As features are usually extracted from the first layer after convolutional 
operations, the script mainly extracts features from `res5c` layer (as named in the author's original [Caffe code](http://ethereon.github.io/netscope/#/gist/d38f3e6091952b45198b)).

Here is a sample code of using the script.
```python
    import torch
    import torch.autograd as A
    import torchvision.models as M

    import res5c
    
    # ResNet18 uses BasicBlock as the building block.
    resnet18 = M.resnet18(pretrained=True)

    # ResNet152 uses Bottleneck as the building block.
    resnet50 = M.resnet50(pretrained=True)

    resnet18_fe = res5c.ResNetFeatureExtractor(resnet18, feat_layer="res5c")
    resnet50_fe = res5c.ResNetFeatureExtractor(resnet50, feat_layer="res5c")

    x = A.Variable(torch.randn(1, 3, 224, 224))
    h18 = resnet18_fe(x)
    h50 = resnet50_fe(x)

    assert h18.size() == (1, 512, 7, 7)
    assert h50.size() == (1, 2048, 7, 7)
```