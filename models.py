import torch.nn as nn
import torch


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

cpu = torch.device('cpu')
device = cpu
if torch.cuda.is_available():
    device = torch.device('cuda')


def _gen_M(X, degree, mode='normal'):
    hidden_size =  sum([i+1 for i in range(degree)])*2
    out_size = 3

    if mode == 'prim':
        out_size += 1
        hidden_size = int(hidden_size*1.4)

    return nn.Sequential(*[
        nn.Linear(len(X[0]), hidden_size)
        ,nn.ReLU()
        ,nn.Linear(hidden_size, hidden_size)
        ,nn.ReLU()
        ,nn.Linear(hidden_size, hidden_size)
        ,nn.ReLU()
        ,nn.Linear(hidden_size,out_size)
    ]).to(device).train()

def _gen_C(X,degree):
    hidden_size =  sum([i+1 for i in range(degree)])*2
    return nn.Sequential(*[
        nn.Linear(len(X[0])+3, hidden_size)
        ,nn.ReLU()
        ,nn.Linear(hidden_size, hidden_size)
        ,nn.ReLU()
        ,nn.Linear(hidden_size, 1)
    ]).to(device).train()


class M(nn.Module):
    def __init__(self, X, degree):
        super(M, self).__init__()
        self.M = _gen_M(X, degree)

    def forward(self, X): return self.M(X), None


class C(nn.Module):
    def __init__(self, X, degree):
        super(C, self).__init__()
        self.C = _gen_C(X, degree)


    def forward(self, X, Y): return self.C(torch.cat((X, Y), 1))


class MC(nn.Module):
    def __init__(self, X, degree):
        super(MC, self).__init__()
        self.M =  _gen_M(X, degree)
        self.C = _gen_C(X,degree)

    def forward(self, X):
        Y = self.M(X)
        XC = torch.cat((X, Y), 1)
        YC = self.C(XC)

        return Y, YC


class Mprim(nn.Module):
    def __init__(self, X, degree):
        super(Mprim, self).__init__()
        self.Mprim = self.M = _gen_M(X, degree,'prim')

    def forward(self, X):
        Y = self.Mprim(X)
        Y, C = torch.split(Y,[3,1],dim=1)
        return Y, C
