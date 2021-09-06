
import sys
import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Arguments import NLPArgument
from ModelHeaders.rnn import RNNClassifier
from ModelHeaders.cnn import CNNClassifier
from Utilitis.utilitis import get_grad_norm, get_parameter_norm, printProgress

def GenerateNLPProcess(arg:NLPArgument, vocab_size, n_classes)->dict:
    nlp_procs = {}
    if arg.use_rnn:
        rnn_model = RNNClassifier(
            input_size=vocab_size,
            word_vec_size=arg.word_vec_size,
            hidden_size=arg.hidden_size,
            n_classes=n_classes,
            n_layers=arg.layer_number,
            dropout_p=arg.dropout,
        )
        optimizer = optim.Adam(rnn_model.parameters())
        crit = nn.NLLLoss()
        if arg.gpu_id >= 0:
            rnn_model.cuda(arg.gpu_id)
            crit.cuda(arg.gpu_id)

        nlp_procs['rnn'] = (rnn_model, optimizer, crit)

    if arg.use_cnn:
        cnn_model = CNNClassifier(
            input_size=vocab_size,
            word_vec_size=arg.word_vec_size,
            n_classes=n_classes,
            use_batch_norm=arg.use_batch_norm,
            dropout_p=arg.dropout,
            window_sizes=arg.window_sizes,
            n_filters=arg.filter_sizes,
        )
        
        optimizer = optim.Adam(cnn_model.parameters())
        crit = nn.NLLLoss()
        if arg.gpu_id >= 0:
            cnn_model.cuda(arg.gpu_id)
            crit.cuda(arg.gpu_id)

        nlp_procs['cnn'] = (cnn_model, optimizer, crit)

    return nlp_procs


def NLPTrainer(model, optimizer, crit, loaders, arg:NLPArgument, modelname):
    best_loss:float = np.inf
    best_model = None
    
    device = next(model.parameters()).device

    iter_progress_freq = int(100)
    train_batch_maxnum = len(loaders.train_loader)
    valid_batch_maxnum = len(loaders.valid_loader)
    
    for e in range(arg.epochs):
        print('\n{} current epoch => {}/{}'.format(modelname, e+1, arg.epochs))
        
        for index, mini_batch in enumerate(loaders.train_loader):
            model.train()
            optimizer.zero_grad()

            x = mini_batch.text[:,:arg.max_length]
            y = mini_batch.label
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = crit(y_hat, y)

            loss.backward()

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

            p_norm = float(get_parameter_norm(model.parameters()))
            g_norm = float(get_grad_norm(model.parameters()))

            # Take a step of gradient descent.
            optimizer.step()
            if int(index % (train_batch_maxnum / iter_progress_freq)) == 0 or index+1 == train_batch_maxnum:
                fix = 'trainning ==>{}/{}   loss = {} , accuracy = {}, |p_norm| = {}, |g_norm| = {}'.format(index, train_batch_maxnum, float(loss), float(accuracy), p_norm, g_norm)
                printProgress(index, train_batch_maxnum, prefix=fix)

        validate_loss:float = np.inf
        for index, mini_batch in enumerate(loaders.valid_loader):
            model.eval()
            with torch.no_grad():
                x = mini_batch.text[:,:arg.max_length]
                y = mini_batch.label
                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                validate_loss = crit(y_hat, y)

                

                if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                    accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
                else:
                    accuracy = 0

                if int(index % (valid_batch_maxnum / iter_progress_freq)) == 0 or index+1 == valid_batch_maxnum:
                    fix = 'valide ==>{}/{}   loss = {} , accuracy = {}, |p_norm| = {}, |g_norm| = {}'.format(index, valid_batch_maxnum, float(loss), float(accuracy), p_norm, g_norm)
                    printProgress(index, valid_batch_maxnum, prefix=fix)

        if validate_loss < best_loss:
            best_model = deepcopy(model.state_dict())
            best_loss = validate_loss

    model.load_state_dict(best_model)
    return model