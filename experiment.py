from copy import deepcopy
import pprint
import json

import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

from gen_data import gen_a, gen_b, gen_c
from models import M, Mprim, MC, C
from evaluate import evaluate

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

cpu = torch.device('cpu')
device = cpu
if torch.cuda.is_available():
    device = torch.device('cuda')

#mode = 'linear'
#mode = 'polynomial'
mode = 'polynomial_w_coef'

def acc_f(Yh,Y,confidences=None):
    acc_vector = []
    if confidences is None:
        for i in range(len(Yh)):
            acc_vector.append(Yh[i].index(max(Yh[i])) == Y[i])

    return np.mean(acc_vector)

def validate(mode, data, net, full_val=False, cnet=None):
    net = net.eval()
    with torch.no_grad():
        X_test, Y_test = data
        test_ds = DataSource(X_test,Y_test,mode)
        test_dl = DataLoader(test_ds, batch_size=200, shuffle=False)
        Yh = []
        Yc = []

        for X, _ in test_dl:
            X = X.to(device)
            _Yh, _Yc = net(X)
            Yh.extend(_Yh.tolist())
            if cnet is not None:
                Yc.extend(cnet(X, _Yh).tolist())
            elif _Yc is None:
                Yc = None
            else:
                Yc.extend(_Yc.tolist())


        if full_val:
            return acc_f(Yh, Y_test), evaluate(Yh, Y_test, Yc)
        else:
            return acc_f(Yh, Y_test)

class DataSource(Dataset):
    def __init__(self, X, Y, mode):
        # One hot encode Y
        Yn = []
        for y in Y:
            Yn.append([0,0,0])
            Yn[-1][y] = 1
        self.Y = torch.Tensor(Yn)

        X = torch.FloatTensor(X)
        if mode in ['polynomial', 'polynomial_w_coef']:
            self.X = torch.log(X)
        elif mode == 'linear':
            self.X = X


    def __len__(self): return len(self.Y)

    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


with open('Results_batch2.txt', 'w') as fp:
    fp.write('')

M_train_time = {}
exp_result = {}

#
for degree in [3,4,5,6]:
    for mode in ['polynomial_w_coef','linear', 'polynomial']:
        for model in ('M','M+C','MC','Mprim'):
            for data_gen_func in [gen_a, gen_b, gen_c]:
                if  model != 'M':
                    max_train_time = max(M_train_time[str(data_gen_func)] * 4,300)
                    print(f'Maximum training time: {max_train_time} !')

                X, Y = data_gen_func(mode, 'train',degree)
                data_validate = data_gen_func(mode, 'validate',degree)

                if model == 'M':
                    net = M(X, degree)
                elif model == 'M+C':
                    net = M(X, degree)
                    cnet = C(X, degree)
                elif model == 'MC':
                    net = MC(X, degree)
                elif model == 'Mprim':
                    net = Mprim(X, degree)

                lr = 3e-3
                ds = DataSource(X,Y,mode)
                dl = DataLoader(ds, batch_size=10, shuffle=True)

                criterion = nn.CrossEntropyLoss()
                def unreduced_criterion(Yh,Y): return (Yh.max(1).indices == Y.max(1).indices).float()
                c_criterion = nn.MSELoss()

                optimizer = torch.optim.SGD(net.parameters(), lr=lr)
                if model == 'M+C':
                    c_optimizer = torch.optim.SGD(cnet.parameters(), lr=lr)

                validation_acc_arr = [0]
                validation_acc_delta_arr = []
                validation_acc_delta_hist_arr = []
                best_validation_acc = 0
                best_model = None
                delta_stop_size = 10
                best_stop_size = delta_stop_size * 5
                lr_reductions = 0
                nr_epochs = 0

                start_train = time.time()
                for epoch in range(3000):
                    nr_epochs += 1
                    if model != 'M':
                        if time.time() - start_train > max_train_time:
                            break

                    if epoch == 0:
                        validation_acc = validate(mode, data_validate, net)
                        validation_acc_pct = round(validation_acc*100,2)
                        print(f'Initial (random weights) validation accuracy of {validation_acc_pct}%')

                    net = net.train()
                    running_loss = 0
                    for X,Y in dl:
                        optimizer.zero_grad()
                        if model == 'M+C':
                            c_optimizer.zero_grad()

                        X = X.to(device)
                        Y = Y.to(device)

                        Yh, Yc = net(X)
                        loss = criterion(Yh,Y.max(1).indices)

                        if Yc is not None:
                            unreduced_loss = unreduced_criterion(Yh,Y).unsqueeze(1)
                            C_loss = c_criterion(Yc, unreduced_loss)
                            C_loss.backward(retain_graph=True)
                        elif model == 'M+C':
                            Yc = cnet(X.detach(), Yh.detach())
                            unreduced_loss = unreduced_criterion(Yh,Y).unsqueeze(1)
                            C_loss = c_criterion(Yc, unreduced_loss)
                            C_loss.backward()
                            c_optimizer.step()

                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()

                    # Stopping logic
                    avg_loss = round(running_loss/len(dl),3)
                    validation_acc = validate(mode, data_validate, net)
                    if validation_acc > best_validation_acc:
                        best_validation_acc = validation_acc
                        best_model = deepcopy(net).to(cpu)

                        validation_acc_pct = round(validation_acc*100,2)
                        print(f'Best validation accuracy of {validation_acc_pct}% for epoch {epoch}')
                        if validation_acc > 0.99:
                            break

                    if epoch % 8 == 0:
                        validation_acc_pct = round(validation_acc*100,2)
                        print(f'Validation accuracy of {validation_acc_pct}% for epoch {epoch}')

                    mean_validation_acc = np.mean(validation_acc_arr)
                    validation_acc_delta = validation_acc - mean_validation_acc

                    validation_acc_arr.append(validation_acc)
                    validation_acc_delta_arr.append(validation_acc_delta)
                    validation_acc_delta_hist_arr.append(validation_acc_delta)

                    validativalidation_acc_arron_acc = validation_acc_arr[-delta_stop_size:]
                    validation_acc_delta_arr = validation_acc_delta_arr[-delta_stop_size:]

                    best_stop =  len(validation_acc_delta_hist_arr) >= best_stop_size and round(np.max(validation_acc_delta_hist_arr[-best_stop_size:]),6) < round(best_validation_acc,6)

                    delta_stop = len(validation_acc_delta_arr) == delta_stop_size and np.mean(validation_acc_delta_arr) < 0

                    if delta_stop:
                        break

                training_time = time.time() - start_train
                if model == 'M':
                     M_train_time[str(data_gen_func)] = training_time

                data_test = data_gen_func(mode, 'test', degree)

                if model == 'M+C':
                    acc, eval_data = validate(mode, data_test,best_model.to(device), True, cnet)
                else:
                    acc, eval_data = validate(mode, data_test,best_model.to(device), True)

                eval_data['8. Training time'] = training_time
                eval_data['9. Nr Epochs'] = nr_epochs

                dataset = str(data_gen_func).split('function ')[1].split(' at')[0].replace('gen_', '')
                acc = round(acc*100,2)
                out = ''
                out += f'Model: {model}'
                out += f'\nDegree: {degree}'
                out += f'\nMode: {mode}'
                out += f'\nDataset: {dataset})'
                out += f'\nAccuracy score of {acc}%'
                out += '\n\n---------------------------\n\n'
                for k,v in eval_data.items():
                    if v is not None:
                        v = round(v,3)
                    out += f'{k} : {v}\n'
                out += '\n---------------------------\n\n'
                print(out)
                with open('Results_batch2.txt', 'a') as fp:
                    fp.write(out)

                if degree not in exp_result: exp_result[degree] = {}
                if mode not in exp_result[degree]: exp_result[degree][mode] = {}
                if model not in exp_result[degree][mode]: exp_result[degree][mode][model] = {}

                exp_result[degree][mode][model][dataset] = eval_data

    json.dump(exp_result, open(f'Results_{degree}_batch2.json', 'w'))
