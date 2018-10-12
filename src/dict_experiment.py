import torch
class D:
    def __init__(self,):
        self.t={'a':torch.ones(size=(2,2)),}
        self.de=2

    def dict(self,inputx,index,epoch):
        x={'aa':self.t['a'][index][epoch],
            'xx':inputx}
        

        return [x]

    def change(self,x):
        for i in x:
            i['aa'] +=  4
            i['aa'] = i['aa'] + 10
a=D()

a.change(a.dict(3,1,1))
print(a.t)

a=[3]
p=a[0]
p += 1
print(a)