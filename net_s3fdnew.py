import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale=1.0):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.weight.data *= 0.0
        self.weight.data += self.scale

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = x / norm * self.weight.view(1,-1,1,1)
        return x

class s3fd(nn.Module):
    def __init__(self, num_classes):
        super(s3fd, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc6     = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
        self.fc7     = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.fc_1 = nn.Linear(2304,num_classes)

        self.conv3_3_norm = L2Norm(256,scale=10)
        self.conv4_3_norm = L2Norm(512,scale=8)
        self.conv5_3_norm = L2Norm(512,scale=5)

       
        self.conv3_3_norm_mbox_conf = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv3_3_norm_mbox_loc  = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_loc  = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_gender = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.l4_3 = nn.Linear(2048,50)
        self.l4_f = nn.Linear(50,num_classes)

        self.conv5_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_loc  = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_gender = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.l5_3 = nn.Linear(2048,50)
        self.l5_f = nn.Linear(50,num_classes)


        self.fc7_mbox_conf     = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        self.fc7_mbox_loc      = nn.Conv2d(1024, 4, kernel_size=3, stride=1, padding=1)
        self.convfc7_norm_gender = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        self.lfc7 = nn.Linear(288,50)
        self.lfc7f= nn.Linear(50,num_classes)

        self.conv6_2_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_loc  = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv6_2_norm_gender = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.l6_2 = nn.Linear(288,50)
        self.l6_f = nn.Linear(50,num_classes)
#         sh= self.get_shape(self.conv4_3_norm_gender)
#         self.linear6_2 = nn.Linear(sh,num_classes)
        
        self.conv7_2_mbox_conf = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_loc  = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        
        # generate input sample and forward to get shape
    def get_shape(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h)); f3_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h)); f4_3 = h
        f4_3 = self.conv4_3_norm(f4_3)
        
        
        gen2 = self.conv4_3_norm_gender(f4_3)
        cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        g2= F.relu(gen2)
        m = g2.view(1,-1)
        ge2 = self.l4_3(m)
        ge2= self.l4_f(ge2)
        

        

        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h)); f5_3 = h
        f5_3 = self.conv5_3_norm(f5_3)
        gen3 = self.conv5_3_norm_gender(f5_3)
        g3= F.relu(gen3)
        m = g2.view(1,-1)
        ge3 = self.l5_3(m)
        ge3= self.l5_f(ge3)
        
        cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        h = F.max_pool2d(h, 2, 2)

        
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h));     ffc7 = h
        gen4 = self.convfc7_norm_gender(ffc7)
        g4= F.relu(gen4)
        m = g4.view(1,-1)
        ge4 = self.lfc7(m)
        ge4= self.lfc7f(ge4)
        cls4 = self.fc7_mbox_conf(ffc7)
        
        
        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h)); f6_2 = h
        gen5 = self.conv6_2_norm_gender(f6_2)
        
        g5= F.relu(gen5)
        m = g4.view(1,-1)
        ge5 = self.l6_2(m)
        ge5= self.l6_f(ge5)
        
        cls5 = self.conv6_2_mbox_conf(f6_2)
        
        h = F.relu(self.conv7_1(h))
        h = F.relu(self.conv7_2(h)); f7_2 = h
        #print(f7_2.size())
        #m = F.max_pool2d(h, 2, 2)
        #print(h.size())
        m = f7_2.view(1,-1)
        #print(m)
        op = self.fc_1(m)
        return [cls2,gen2,cls3,gen3,cls4,gen4,cls5,gen5]

