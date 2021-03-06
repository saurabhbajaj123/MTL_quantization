import numpy as np
from sys import exit
import torch

class Trainer():
    def __init__(self, model, tasks, train_dataloader, val_dataloader, criterion_dict, metric_dict, 
                 lr=0.001, decay_lr_freq=4000, decay_lr_rate=0.5,
                 print_iters=50, val_iters=200, save_iters=200):
        print("initializing the Trainer")
        super(Trainer, self).__init__()
        self.model = model
        self.startIter = 0
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_lr_freq, gamma=decay_lr_rate)
        
        self.tasks = tasks
        
        self.train_dataloader = train_dataloader
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = val_dataloader
        self.criterion_dict = criterion_dict
        self.metric_dict = metric_dict
        
        self.loss_list = {}
        self.set_train_loss()
        
        self.print_iters = print_iters
        self.val_iters = val_iters
        self.save_iters = save_iters
    
    def train(self, iters, loss_lambda, savePath=None, reload=None):
        self.model.train()
        if reload is not None and reload != 'false' and savePath is not None:
            self.load_model(savePath, reload)

        for i in range(self.startIter, iters):
            print('i = {}'.format(i))
            self.train_step(loss_lambda)

            if (i+1) % self.print_iters == 0:
                self.print_train_loss(i)
                self.set_train_loss()
            if (i+1) % self.val_iters == 0:
                self.validate(i)
            if (i+1) % self.save_iters == 0:
                if savePath is not None:
                    self.save_model(i, savePath)
            
        # Reset loss list and the data iters
        self.set_train_loss()
        return
    
    def train_step(self, loss_lambda):
        print("starting train step")
        self.model.train()
        try:
            data = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            data = next(self.train_iter)
            
        x = data['input']#.cuda()
        # print("size(x) = ", x.size())
        self.optimizer.zero_grad()
        output = self.model(x)
        # print("output = {}".format(output.size()))
        loss = 0
        for task in self.tasks:
            print("task = {}".format(task))
            y = data[task]#.cuda()
            if task + '_mask' in data:
                tloss = self.criterion_dict[task](output[task], y, data[task + '_mask'])#.cuda())
            else:
                tloss = self.criterion_dict[task](output[task], y)
                
            self.loss_list[task].append(tloss.item())
            loss += loss_lambda[task] * tloss
            print("loss = {}".format(loss))
        self.loss_list['total'].append(loss.item())
        
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return
    
    def validate(self, it):
        self.model.eval()
        loss_list = {}
        for task in self.tasks:
            loss_list[task] = []
        
        for i, data in enumerate(self.val_dataloader):
            x = data['input']#.cuda()
            output = self.model(x)

            for task in self.tasks:
                y = data[task]#.cuda()
                if task + '_mask' in data:
                    tloss = self.criterion_dict[task](output[task], y, data[task + '_mask'])#.cuda())
                    self.metric_dict[task](output[task], y, data[task + '_mask'])#.cuda())
                else:
                    tloss = self.criterion_dict[task](output[task], y)
                    self.metric_dict[task](output[task], y)
                loss_list[task].append(tloss.item())

        for task in self.tasks:
            val_results = self.metric_dict[task].val_metrics()
            print('[Iter {} Task {}] Val Loss: {:.4f}'.format((it+1), task[:4], np.mean(loss_list[task])), flush=True)
            print(val_results, flush=True)
        print('======================================================================', flush=True)
        return
    
    # helper functions
    def set_train_loss(self):
        for task in self.tasks:
            self.loss_list[task] = []
        self.loss_list['total'] = []
        return
    
    def load_model(self, savePath, reload):
        model_name = True
        for task in self.tasks:
            if task not in reload:
                model_name = False
                break
        if model_name:
            state = torch.load(savePath + reload)
            self.startIter = state['iter'] + 1
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
        else:
            print('Cannot load from models trained from different tasks.')
            exit()
        return
    
    def save_model(self, it, savePath):
        state = {'iter': it,
                'state_dict': self.model.state_dict(),
                'layout': self.model.layout,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()}
        if hasattr(self.model, 'branch') and self.model.branch is not None:
            torch.save(state, savePath + '_'.join(self.tasks) + '_b' + str(self.model.branch) + '.model')
        elif hasattr(self.model, 'layout') and self.model.layout is not None:
            torch.save(state, savePath + '_'.join(self.tasks) + '.model')
        return
    
    def print_train_loss(self, it):
        # Function: Print loss for each task
        for task in self.tasks:
            if self.loss_list[task]:
                avg_loss = np.mean(self.loss_list[task])
            else:
                continue
            print('[Iter {} Task {}] Train Loss: {:.4f}'.format((it+1), task[:4], avg_loss), flush=True)
        print('[Iter {} Total] Train Loss: {:.4f}'.format((it+1), np.mean(self.loss_list['total'])), flush=True)
        print('======================================================================', flush=True)
        return