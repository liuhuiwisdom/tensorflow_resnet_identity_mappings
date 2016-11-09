# Residual Neural Network with Identity Mappings

Residual neural network with identity mappings as described in paper:
https://arxiv.org/abs/1603.05027

resnet34_im_1000_classes.py: ImageNet ready ResNet34 with identity mappings. Could be easily extended to any number of layers.
abstractions.py: contains building blocks for ResNet with identity mappings. 
- residual_block_im - for 34-layer
- residual_block_deep_im - for 110, 164, 1001-layer
