import torch
#TODO: get rid of transpose#
import torch.nn as nn
import torch.nn.functional as F

#______Regression Model________#
# This models i use to predict several biological parameters
# At the very beggining i tried to validate my GAN output with trained regressors
# However this models was not precision enough for my needs

# Also i used several Reg. models each for each Biological parametr
# Here i keep the most precision and universal model

class DNA_2_Bio_Regression(nn.Module):
    '''
        The most precision and universal regression model 
        to predict Biological parametr with DNA sequence
    '''
    def __init__(self,seq_size):
        super(DNA_2_Bio_Regression,self).__init__()
        self.conv1 = nn.Conv1d(1 ,200,5,1, padding = 'same')
        self.conv2 = nn.Conv1d(200 ,200,5,1, padding = 'same')
        self.conv3 = nn.Conv1d(200 ,1,5,1, padding = 'same')
        self.pool = nn.MaxPool1d(51,1,25)

        self.lin1 = nn.Linear(seq_size*5, seq_size)
        self.lin2 = nn.Linear(seq_size, int(seq_size/4))
        self.final = nn.Linear(int(seq_size/4), 1)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(seq_size*5)

    def forward(self, x): #convert + flatten
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.conv3(x)
        #x = self.relu(self.lin(x))
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.final(x)
        return x

#______GAN Models________#
# Here is GAN architecture that i used to augmentate dataset
# with sintetic DNA sequences

class BasicBlock_1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock_1D, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out
        
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out
        
class DNA_Seq_Generator(nn.Module):
    '''
        DNA Sequence generator which has to work
        in pair with `DNA_Seq_Critic()` 
    '''
    def __init__(self, gen_dim = 100, max_seq_len = 1200, annotated=False, res_layers=5):
        super(DNA_Seq_Generator, self).__init__()
        self.gen_dim = gen_dim
        self.block1 = BasicBlock_1D(gen_dim ,gen_dim)
        self.block2 = BasicBlock_1D(gen_dim ,gen_dim)
        self.block3 = BasicBlock_1D(gen_dim ,gen_dim)
        self.conv1 = nn.Sequential(
                        nn.Conv1d(100,5, kernel_size=1),
                        nn.ReLU())
        
        self.softmax = nn.Softmax(dim = 1)
        self.linear = nn.Linear(5,100)
        

    def forward(self,x):
        out = self.linear(x) 
        out = self.block1(out.view(1,100,-1))
        out = self.block2(out)
        out = self.block3(out)
        out = self.softmax(self.conv1(out))
        return out
    
class DNA_Seq_Critic(nn.Module):
    '''
        DNA Sequence critic which has to work
        in pair with `DNA_Seq_Generator()` 
    '''
    def __init__(self, gen_dim = 100, max_seq_len = 1200, num_channels=100  , res_layers=5):
        super(DNA_Seq_Critic, self).__init__()
        self.block1 = BasicBlock_1D(gen_dim ,gen_dim)
        self.block2 = BasicBlock_1D(gen_dim ,gen_dim)
        self.block3 = BasicBlock_1D(gen_dim ,gen_dim)
        self.conv1 = nn.Sequential(
                        nn.Conv1d(5,100, kernel_size=1),
                        nn.ReLU())
        
        self.linear = nn.Linear(100*max_seq_len,max_seq_len)

    def forward(self,x):
        out = self.conv1(x)
        out = self.block1(out.view(1,100,-1))
        out = self.block2(out)
        out = self.block3(out)
        out = self.linear(out.flatten())
        return out


    