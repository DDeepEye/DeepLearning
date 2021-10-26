import sys
import os

import torch

from train import main
from train import define_argparser

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Arguments import NMTArgumets

def overwrite_config(config, prev_config):
    # This method provides a compatibility for new or missing arguments.
    for prev_key in vars(prev_config).keys():
        if not prev_key in vars(config).keys():
            # No such argument in current config. Ignore that value.
            print('WARNING!!! Argument "--%s" is not found in current argument parser.\tIgnore saved value:' % prev_key,
                  vars(prev_config)[prev_key])

    for key in vars(config).keys():
        if not key in vars(prev_config).keys():
            # No such argument in saved file. Use current value.
            print('WARNING!!! Argument "--%s" is not found in saved model.\tUse current value:' % key,
                  vars(config)[key])
        elif vars(config)[key] != vars(prev_config)[key]:
            if '--%s' % key in sys.argv:
                # User changed argument value at this execution.
                print('WARNING!!! You changed value for argument "--%s".\tUse current value:' % key,
                      vars(config)[key])
            else:
                # User didn't changed at this execution, but current config and saved config is different.
                # This may caused by user's intension at last execution.
                # Load old value, and replace current value.
                vars(config)[key] = vars(prev_config)[key]

    return config


def continue_main(config, args :NMTArgumets, main):
    config.model_fn = args.model_filepath    
    # If the model exists, load model and configuration to continue the training.
    if os.path.isfile(config.model_fn):
        saved_data = torch.load(config.model_fn, map_location='cpu')
        prev_config = saved_data['config']

        config.rl_lr = args.rl_lr
        config.rl_n_samples = args.rl_n_samples        
        config.rl_n_gram = args.rl_n_gram
        config.rl_reward = args.rl_reward
        config.rl_batch_ratio = args.rl_batch_ratio

        config = overwrite_config(config, prev_config)

        if args.init_epoch > -1 :
            config.init_epoch = args.init_epoch
        if args.epochs > -1 and args.epochs > args.init_epoch:
            config.n_epochs = args.epochs 

        if args.rl_init_epoch > -1:
            config.rl_init_epoch = args.rl_init_epoch
        if args.rl_n_epochs > -1 and args.rl_n_epochs > args.rl_init_epoch:
            config.rl_n_epochs = args.rl_n_epochs

        config.max_grad_norm = args.max_grad_norm

        model_weight = saved_data['model']
        opt_weight = saved_data['opt']

        cur_dir = os.path.dirname(__file__)
        if len(cur_dir) == 0 : cur_dir = '.'

        if len(args.train_filepath) > 0:
            config.train = args.train_filepath
        else:
            config.train = cur_dir+config.train.strip('.')

        if len(args.valid_filepath) > 0:
            config.valid = args.valid_filepath
        else:
            config.valid = cur_dir+config.valid.strip('.')

        main(config, model_weight=model_weight, opt_weight=opt_weight, src_vocab=saved_data['src_vocab'], tgt_vocab=saved_data['tgt_vocab'])
    else:
        print('Cannot find file %s' % config.model_fn)

if __name__ == '__main__':
    
    # args = NMTArgumets(model_filepath='NMT_2021_0915.pth'
    #                 , train_filepath=''
    #                 , valid_filepath=''
    #                 , init_epoch= -1
    #                 , epochs=25
    #                 )

    # args = NMTArgumets(model_filepath='./2021.0919.NMT/NMT.pth'
    #                 , train_filepath=''
    #                 , valid_filepath=''
    #                 , init_epoch= -1
    #                 , epochs=38
    #                 , is_shutdonw=True
    #                 )

    cur_dir = os.path.dirname(__file__)
    if len(cur_dir) == 0 : cur_dir = '.'
    args = NMTArgumets(model_filepath=cur_dir+'/2021.1004.TFM/Transformer.pth'
                    , train_filepath=cur_dir+'/corpus/corpus.train.tr'
                    , valid_filepath=cur_dir+'/corpus/corpus.valid.tr'
                    , init_epoch= -1
                    , epochs=41
                    , rl_n_epochs=2
                    , rl_batch_ratio=0.4
                    , is_shutdonw=True
                    )

    config = define_argparser(is_continue=True, args = args)
    continue_main(config, args, main)
    if args.is_shutdon:os.system('shutdown -s -f')
    