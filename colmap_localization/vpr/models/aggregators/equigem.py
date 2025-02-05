import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn
import torch.nn.functional as F
    
class EquivariantGeMPool(nn.Module):
    """
    Equivariant Generalized Mean Pooling (GeM) for an equivariant framework like e2cnn.
    This implementation ensures that the pooling respects the group's symmetries and uses
    power mean pooling concepts.
    """
    def __init__(self, in_channels,out_channels, p=3, eps=1e-6,N=8):
        super().__init__()
        
        # Assuming the input features use a trivial representation
        # self.field_type = enn.FieldType(self.group, [self.group.trivial_repr()])
        self.group=gspaces.rot2dOnR2(N=N)
        self.field_type = enn.FieldType(self.group, in_channels*[self.group.regular_repr])
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.avg_pool = enn.PointwiseAvgPool2D(self.field_type,(8,8))
        self.gpool = enn.GroupPooling(self.field_type)
        self.norm = enn.NormPool(self.field_type)
        # self.field_type_in = enn.FieldType(gspaces., in_channels*[self.group.regular_repr])
        # self.field_type_out = enn.FieldType(self.group, out_channels*[self.group.regular_repr])
        self.fc = nn.Linear(in_channels,out_channels)#enn.Linear(self.field_type,self.field_type_out)

    def forward(self, x):
        # x=self.norm(x)
        # Wrap the input tensor in a GeometricTensor
        # x = enn.GeometricTensor(x, self.field_type)
        # Apply the power operation, clamp to avoid numerical issues
        x.tensor = x.tensor.clamp(min=self.eps).pow(self.p)
        # Use adaptive average pooling to respect the input size and symmetry
        x = self.avg_pool(x)
        x.tensor=x.tensor.pow(1./self.p)
        # Flatten and normalize
        
        x = self.gpool(x)
        x = x.tensor.flatten(1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)
    

if __name__ == '__main__':
    
    class GeMPool(nn.Module):
        """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
        we add flatten and norm so that we can use it as one aggregation layer.
        """
        def __init__(self, p=3, eps=1e-6):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1)*p)
            self.eps = eps

        def forward(self, x):
            x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
            x = x.flatten(1)
            return F.normalize(x, p=2, dim=1)
        

    # Example of using EquivariantGeMPool
    N=4
    group = gspaces.rot2dOnR2(N=N)
    type_in = enn.FieldType(group, 800*[group.regular_repr])
    model = EquivariantGeMPool(N=N)
    
    input = torch.randn(1, 800*N, 8, 8)
    # rotate the input tensor
    input_rot = torch.rot90(input, 1, [2, 3])

    x = enn.GeometricTensor(input, type_in)
    out = model(x)

    x = enn.GeometricTensor(input_rot, type_in)
    outr = model(x)
    print('\nWith equiGeM:',torch.norm(out-outr))

    # Using the default Equivision
    x = enn.GeometricTensor(input, type_in)
    x = enn.PointwiseAdaptiveAvgPool2D(type_in, (1, 1))(x)
    x = enn.GroupPooling(type_in)(x).tensor.squeeze(-2).squeeze(-1)
    out = F.normalize(x, p=2, dim=1)
    
    x = enn.GeometricTensor(input_rot, type_in)
    x = enn.PointwiseAdaptiveAvgPool2D(type_in, (1, 1))(x)
    x = enn.GroupPooling(type_in)(x).tensor.squeeze(-2).squeeze(-1)
    outr = F.normalize(x, p=2, dim=1)
    print('With default Equivision:',torch.norm(out-outr))

    #Using the normal GeMPool
    model = GeMPool()
    out = model(input)
    outr = model(input_rot)
    print('With GeMPool:',torch.norm(out-outr))



    # print(out.shape)
