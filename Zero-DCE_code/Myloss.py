import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)


        return k

			
class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.tensor([1.], device=org_pool.device) + 10000*torch.min(org_pool - torch.tensor([0.3], device=org_pool.device),torch.tensor([0.], device=org_pool.device)),torch.tensor([0.5], device=org_pool.device))
        E_1 = torch.mul(torch.sign(enhance_pool - torch.tensor([0.5], device=org_pool.device)) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.tensor([self.mean_val], device=mean.device),2))
        return d
        
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)
    def forward(self, x ):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b,c,h,w = x.shape
        # x_de = x.cpu().detach().numpy()
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        # print(k)
        

        k = torch.mean(k)
        return k

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3


# ---------------------------------------------------------------------------
# Loss terms written to follow the equations in the Zero-DCE paper
# (Guo et al., CVPR 2020, Eqs. 4-7). The classes above are the released
# versions and are left unchanged; these are used only with --paper_losses.
# Each returns a scalar averaged over the batch.
# ---------------------------------------------------------------------------

class L_spa_paper(nn.Module):
    """Eq. 4: (1/K) sum_i sum_{j in Omega(i)} (|Y_i - Y_j| - |I_i - I_j|)^2,
    over 4x4 regions of the channel-averaged image, Omega(i) = 4 neighbours."""

    def __init__(self, region=4):
        super(L_spa_paper, self).__init__()
        self.pool = nn.AvgPool2d(region)

    def forward(self, org, enhance):
        I = self.pool(torch.mean(org, 1, keepdim=True))
        Y = self.pool(torch.mean(enhance, 1, keepdim=True))
        # Each adjacent pair (i, j) appears twice in the double sum (j in Omega(i)
        # and i in Omega(j)); regions on the border simply have fewer neighbours.
        d_v = (torch.abs(Y[:, :, 1:, :] - Y[:, :, :-1, :]) - torch.abs(I[:, :, 1:, :] - I[:, :, :-1, :])) ** 2
        d_h = (torch.abs(Y[:, :, :, 1:] - Y[:, :, :, :-1]) - torch.abs(I[:, :, :, 1:] - I[:, :, :, :-1])) ** 2
        K = I.shape[2] * I.shape[3]
        per_image = 2 * (d_v.sum(dim=(1, 2, 3)) + d_h.sum(dim=(1, 2, 3))) / K
        return per_image.mean()


class L_exp_paper(nn.Module):
    """Eq. 5: (1/M) sum_k |Y_k - E| over non-overlapping 16x16 regions."""

    def __init__(self, patch_size=16, mean_val=0.6):
        super(L_exp_paper, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        Y = self.pool(torch.mean(x, 1, keepdim=True))
        return torch.mean(torch.abs(Y - self.mean_val))


class L_color_paper(nn.Module):
    """Eq. 6: sum over (R,G), (R,B), (G,B) of (J^p - J^q)^2, J = channel mean."""

    def forward(self, x):
        mr, mg, mb = torch.split(torch.mean(x, [2, 3]), 1, dim=1)
        per_image = (mr - mg) ** 2 + (mr - mb) ** 2 + (mg - mb) ** 2
        return per_image.mean()


class L_TV_paper(nn.Module):
    """Eq. 7: (1/N) sum_n sum_c (|grad_x A_n^c| + |grad_y A_n^c|)^2, N = iterations.
    The paper does not say how pixels are combined; we average over pixels.
    Forward differences are cropped to a common (H-1) x (W-1) grid."""

    def __init__(self, n_iter=8):
        super(L_TV_paper, self).__init__()
        self.n_iter = n_iter

    def forward(self, A):
        gx = torch.abs(A[:, :, :-1, 1:] - A[:, :, :-1, :-1])
        gy = torch.abs(A[:, :, 1:, :-1] - A[:, :, :-1, :-1])
        per_pixel = ((gx + gy) ** 2).sum(dim=1) / self.n_iter
        return per_pixel.mean()
