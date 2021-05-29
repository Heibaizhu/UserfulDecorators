import torch  
import torch.nn.functional as F
from functools import wraps 
import math 
import pdb 


class SizePadding:
    def __init__(self, factor=8, mode='reflect'):
        self.factor=8
        self.mode = mode
    
    def __call__(self, func):
        @wraps(func)
        def decorated(*args, **kwargs):
            args = list(args)
            input = args[1] # skip arg self
            B, C, H, W = input.shape 
            newH = math.ceil(H / self.factor) * self.factor 
            newW = math.ceil(W / self.factor) * self.factor 
            diffH = newH - H 
            diffW = newW - W 
            input = F.pad(input, [diffH // 2, diffH - diffH // 2, diffW // 2, diffW - diffW // 2])
            args[1] = input 
            output = func(*args, **kwargs)
            if isinstance(output, tuple):
                output = list(output)
                y = output[0]
                y = y[:, :, diffH // 2: newH - diffH + diffH // 2, diffW // 2: newW - diffW + diffW // 2]
                output[0] = y 
            else:
                output = output[:, :, diffH //2: newH - diffH + diffH // 2, diffW // 2: newW - diffW + diffW // 2]
            return output 
        return decorated 


class Model:
    @SizePadding()
    def __call__(self, x):
        B, C, H, W = x.shape 
        assert H % 8 == 0 and W % 8 == 0
        print("x.shape:", x.shape)
        return x , x

if __name__ == '__main__':
    model = Model()
    input = torch.randn(1, 3, 255, 255)
    output = model(input)
