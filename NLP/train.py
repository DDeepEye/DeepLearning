import sys
import os

import torch

from Utilitis.data_loader import DataLoader
from trainer import NLPTrainer , GenerateNLPProcess

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Arguments import NLPArgument

def train_run(arg:NLPArgument):
    loaders = DataLoader(
        train_fn=arg.train_filepath,
        batch_size=arg.batch_size,
        valid_ratio=.1,
        min_freq=arg.min_vocab_freq,
        max_vocab=arg.max_vocab_size,
        device=arg.gpu_id
        )

    print('batch_size = {}, gpu id = {}'.format(arg.batch_size, arg.gpu_id))

    vocab_size = len(loaders.text.vocab)
    n_classes = len(loaders.label.vocab)

    nlp_procs = GenerateNLPProcess(arg, vocab_size=vocab_size, n_classes=n_classes)
    best_models = {}
    for key, value in nlp_procs.items(): 
        print(value[0])
        best_models[key] = NLPTrainer(model = value[0],
                                optimizer = value[1],
                                crit = value[2],
                                loaders = loaders, 
                                arg = arg,
                                modelname=key)

    savedata = {
            'config': arg,
            'vocab': loaders.text.vocab,
            'classes': loaders.label.vocab,}

    for k in nlp_procs.keys():
        savedata[k] = best_models[k].state_dict()

    torch.save(
        savedata,
        arg.model_filepath
    )

if __name__ == '__main__':
    arg = NLPArgument(model_filepath='Models/review.pth', 
                    train_filepath='Data/review.sorted.uniq.refined.tsv',
                    gpu_id=0 ,
                    batch_size=512 ,

                    use_rnn=True, 
                    hidden_size = 128,
                    layer_number = 4,

                    use_cnn=True, 
                    window_sizes=[3,4,5,6,7,8], 
                    filter_sizes=[128,128,128,128,128,128])

    train_run(arg)
                
    