import equivision.models
import equivision.models
import torch
import torch.nn as nn
import torchvision
import numpy as np
import sys
import os
import torchvision
# from models.backbones.reresnet.ReDet.mmdet.models.backbones.re_resnet import ReResNet
# sys.path.append(os.path.relpath(os.path.dirname(os.path.realpath(__file__))))
# from reresnet.re_resnet import ReResNet, FIELD_TYPE, enn
class myResNet(nn.Module):
    def __init__(self,
                 model_name='resnet50',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[],
                 ):
        """Class representing the resnet backbone used in the pipeline
        we consider resnet network as a list of 5 blocks (from 0 to 4),
        layer 0 is the first conv+bn and the other layers (1 to 4) are the rest of the residual blocks
        we don't take into account the global pooling and the last fc

        Args:
            model_name (str, optional): The architecture of the resnet backbone to instanciate. Defaults to 'resnet50'.
            pretrained (bool, optional): Whether pretrained or not. Defaults to True.
            layers_to_freeze (int, optional): The number of residual blocks to freeze (starting from 0) . Defaults to 2.
            layers_to_crop (list, optional): Which residual layers to crop, for example [3,4] will crop the third and fourth res blocks. Defaults to [].

        Raises:
            NotImplementedError: if the model_name corresponds to an unknown architecture. 
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze
    
        if pretrained:
            # the new naming of pretrained weights, you can change to V2 if desired.
            weights = 'IMAGENET1K_V1'
        else:
            weights = None

        if 'swsl' in model_name or 'ssl' in model_name:
            # These are the semi supervised and weakly semi supervised weights from Facebook
            self.model = torch.hub.load(
                'facebookresearch/semi-supervised-ImageNet1K-models', model_name)
        else:
            if 'e2wrn' in model_name:
                import equivision
                if pretrained:
                    
                    self.model = equivision.models.c8resnet50(pretrained=True)
                    # self.model = equivision.models.c8resnet18(pretrained=True)
                
                    # # weights= torch.load('models/backbones/reresnet/re_resnet50_c8_batch256-25b16846.pth')['state_dict']
                    # del weights['head.fc.weight'], weights['head.fc.bias']
                    # self.model = ReResNet(50,with_geotensor=True)
                    # self.model.train()
                    # self.model.load_state_dict(weights)
                    # self.model._freeze_stages(layers_to_freeze)

                
                else:
                    self.model = equivision.models.c8resnet50(pretrained=False)
                    # self.model = equivision.models.c8resnet18(pretrained=False)
                
                                    # in_type = enn.nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
            if 'e2wrn50c8' in model_name:
                
                if pretrained:
                    
                    self.model = equivision.models.c8resnet50(pretrained=True)
                    # self.model = equivision.models.c8resnet18(pretrained=True)
                
                    # # weights= torch.load('models/backbones/reresnet/re_resnet50_c8_batch256-25b16846.pth')['state_dict']
                    # del weights['head.fc.weight'], weights['head.fc.bias']
                    # self.model = ReResNet(50,with_geotensor=True)
                    # self.model.train()
                    # self.model.load_state_dict(weights)
                    # self.model._freeze_stages(layers_to_freeze)

                
                else:
                    self.model = equivision.models.c8resnet50(pretrained=False)
                    # self.model = equivision.models.c8resnet18(pretrained=False)
                
                                    # in_type = enn.nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
            elif 'e2wrn50c4' in model_name:
                if pretrained:
                    self.model = equivision.models.c4resnet50(pretrained=True)
                else:
                    self.model = equivision.models.c4resnet50(pretrained=False)

            elif 'e2wrn18c4' in model_name:
                if pretrained:
                    self.model = equivision.models.c4resnet18(pretrained=True)
                else:
                    self.model = equivision.models.c4resnet18(pretrained=False)
            elif 'e2wrn18c8' in model_name:
                if pretrained:
                    self.model = equivision.models.c8resnet18(pretrained=True)
                else:
                    self.model = equivision.models.c8resnet18(pretrained=False)
 
            # elif 'resnext50' in model_name:
            #     self.model = torchvision.models.resnext50_32x4d(
            #         weights=weights)
            # elif 'resnet50' in model_name:
            #     self.model = torchvision.models.resnet50(weights=weights)
            # elif '101' in model_name:
            #     self.model = torchvision.models.resnet101(weights=weights)
            # elif '152' in model_name:
            #     self.model = torchvision.models.resnet152(weights=weights)
            # elif '34' in model_name:
            #     self.model = torchvision.models.resnet34(weights=weights)
            # elif '18' in model_name:
            #     # self.model = torchvision.models.resnet18(pretrained=False)
            #     self.model = torchvision.models.resnet18(weights=weights)
            # elif 'wide_resnet50_2' in model_name:
            #     self.model = torchvision.models.wide_resnet50_2(
            #         weights=weights)
            else:
                raise NotImplementedError(
                    'Backbone architecture not recognized!')

        # freeze only if the model is pretrained
        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)


        # remove the avgpool and most importantly the fc layer
        # self.model.relu = None
        self.model.fc = None

        if 4 in layers_to_crop:
            self.model.layer4 = None
        if 3 in layers_to_crop:
            self.model.layer3 = None

        # out_channels = 2048
        # if '34' in model_name or '18' in model_name:
        #     out_channels = 512
            
        # self.out_channels = out_channels // 2 if self.model.layer4 is None else out_channels
        # self.out_channels = self.out_channels // 2 if self.model.layer3 is None else self.out_channels

    def forward(self, x):
        # x = self.model.conv1(x)
        # x = self.model.bn1(x)
        # x = self.model.relu(x)
        # x = self.model.maxpool(x)
        # x = self.model.layer1(x)
        # x = self.model.layer2(x)
        # if self.model.layer3 is not None:
        #     x = self.model.layer3(x)
        # if self.model.layer4 is not None:
        #     x = self.model.layer4(x)
        # x=self.model.in_type(x)
        # x= self.model.conv1(x)
        # x= self.model.layer1(x)
        # x= self.model.restrict1(x)
        # x= self.model.layer2(x)
        # x= self.model.restrict2(x)
        # x= self.model.layer3(x)
        # x= self.model.bn(x)
        # x= self.gpool(x).tensor
        # # x= self.pool(x).tensor
        # # x=self.model(x).tensor
        # x = x[:,:,:253,:253]
        # x = torchvision.transforms.Resize(253)(x)
        # x=self.model.in_type(x)
        # x = self.model.conv1(x)

        # x = self.model.layer1(x)
        
        # x = self.model.layer2(self.model.restrict1(x))
        
        # x = self.model.layer3(self.model.restrict2(x))
        
        # x = self.model.bn(x)
        # x = self.model.relu(x)

        # # x= self.model.gpool(x).tensor
        
        # # extract the tensor from the GeometricTensor to use the common Pytorch operations
        # x = x.tensor
        x=self.model.forward_features(x)
        x = self.model.avgpool(x)
        x = self.model.gpool(x).tensor.squeeze(-2).squeeze(-1)

    
        return x
    
if __name__ == '__main__':
    model = myResNet('e2wrn')
    print(model)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)

    model.conv1