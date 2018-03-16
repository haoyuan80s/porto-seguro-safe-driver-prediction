from .importing import *

#__all__ = ['V', 'T', 'N', 'DAE_Net']

def V(x):
    if isinstance(x, np.ndarray):
        return V(T(x))
    elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.cuda.FloatTensor):
        return Variable(x)

def T(x):
    if isinstance(x, torch.FloatTensor) or isinstance(x, torch.cuda.FloatTensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32))

def N(x):
    if isinstance(x, torch.autograd.variable.Variable):
        return x.data.numpy()
    elif isinstance(x, torch.FloatTensor):
        return x.numpy()

class DAE_Net(nn.Module):
    def __init__(self, h_size, gpu = False):
#        super(DAE_Net, self).__init__()
        super(self.__class__, self).__init__()
        self.h_size = h_size
        self.gpu = gpu
        self.l1 = nn.Linear(221,h_size)
        self.l2 = nn.Linear(h_size,h_size)
        self.l3 = nn.Linear(h_size,h_size)
        self.l4 = nn.Linear(h_size,221)
        
    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        return self.l4(h3)
    
    def hidden_feature(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        return torch.cat([h1, h2, h3], dim = -1)
    
    def fit(self,
            x,
            y,
            x_val = None,
            y_val = None,
            epoch = 2,
            batch_size = 64,
            lr = 0.001,
            criterion = nn.MSELoss,
            optimizer = optim.Adam):
        if self.gpu:
            self.cuda()
            x,y = x.cuda(), y.cuda()
            if not x_val is None: x_val, y_val = x_val.cuda(), y_val.cuda()
        logging.info(f"h_size = {self.h_size}, epoch = {epoch}, batch_size = {batch_size}, lr = {lr},  criterion = {criterion},  optimizer = {optimizer}")
        criterion = criterion()
        optimizer = optimizer(self.parameters(), lr=lr)
        data_set = Data.TensorDataset(x, y)
        loader = Data.DataLoader(data_set, batch_size= batch_size, shuffle=True)
        print('Progress:')
        for ep_i in tqdm(range(epoch)):
            optimizer.
            loss_ = 0
            for i, (inp, tar) in tqdm(enumerate(loader)):
                inp, tar = V(inp), V(tar)
                optimizer.zero_grad()
                outp = self(inp) # <=> self.forward(inp)
                loss = criterion(outp, tar)
                loss_ = loss_*(1- 1/(i+1)) + loss.data[0]* (1/(i+1))
                print(f"\r epoch {ep_i}: tr_loss -- {loss_}",end=", ")
                loss.backward()
                optimizer.step()
            if not x_val is None:
                val_loss = criterion(self(V(x_val)), V(y_val)).data[0]
                print(f"val_loss -- {val_loss}")
                logging.info(f"Epoch {ep_i}, tr_loss -- {loss_}, val_loss -- {val_loss}")
