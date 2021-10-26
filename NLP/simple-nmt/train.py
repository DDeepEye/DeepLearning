from re import I
import sys
import os
import argparse
import pprint

import torch
from torch import optim
import torch.nn as nn

import torch_optimizer as custom_optim

from simple_nmt.data_loader import DataLoader
import simple_nmt.data_loader as data_loader

from simple_nmt.models.seq2seq import Seq2Seq
from simple_nmt.models.transformer import Transformer

from simple_nmt.trainer import BaseTrainer
from simple_nmt.rl_trainer import MinimumRiskTrainer

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Arguments import NMTArgumets


def define_argparser(is_continue=False , args:NMTArgumets = None):
    p = argparse.ArgumentParser()

    is_required = (not is_continue and args is None) 

    p.add_argument(
        '--model_fn',
        required=is_required,
        help='Model file name to save. Additional information would be annotated to the file name.'
    )
    
    p.add_argument(
        '--train',
        required=is_required,
        help='Training set file name except the extention. (ex: train.en --> train)'
    )
    p.add_argument(
        '--valid',
        required=is_required,
        help='Validation set file name except the extention. (ex: valid.en --> valid)'
    )
    p.add_argument(
        '--lang',
        required=is_required,
        help='Set of extention represents language pair. (ex: en + ko --> enko)'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s'
    )
    p.add_argument(
        '--off_autocast',
        action='store_true',
        help='Turn-off Automatic Mixed Precision (AMP), which speed-up training.',
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=20,
        help='Number of epochs to train. Default=%(default)s'
    )
    p.add_argument(
        '--verbose',
        type=int,
        default=2,
        help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s'
    )
    p.add_argument(
        '--init_epoch',
        required=is_required,
        type=int,
        default=0,
        help='Set initial epoch number, which can be useful in continue training. Default=%(default)s'
    )

    p.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Maximum length of the training sequence. Default=%(default)s'
    )
    p.add_argument(
        '--dropout',
        type=float,
        default=.2,
        help='Dropout rate. Default=%(default)s'
    )
    p.add_argument(
        '--word_vec_size',
        type=int,
        default=512,
        help='Word embedding vector dimension. Default=%(default)s'
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='Hidden size of LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--n_layers',
        type=int,
        default=4,
        help='Number of layers in LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=5.,
        help='Threshold for gradient clipping. Default=%(default)s'
    )
    p.add_argument(
        '--iteration_per_update',
        type=int,
        default=1,
        help='Number of feed-forward iterations for one parameter update. Default=%(default)s'
    )

    p.add_argument(
        '--lr',
        type=float,
        default=1.,
        help='Initial learning rate. Default=%(default)s',
    )

    p.add_argument(
        '--lr_step',
        type=int,
        default=1,
        help='Number of epochs for each learning rate decay. Default=%(default)s',
    )
    p.add_argument(
        '--lr_gamma',
        type=float,
        default=.5,
        help='Learning rate decay rate. Default=%(default)s',
    )
    p.add_argument(
        '--lr_decay_start',
        type=int,
        default=10,
        help='Learning rate decay start at. Default=%(default)s',
    )

    p.add_argument(
        '--use_adam',
        action='store_true',
        help='Use Adam as optimizer instead of SGD. Other lr arguments should be changed.',
    )
    p.add_argument(
        '--use_radam',
        action='store_true',
        help='Use rectified Adam as optimizer. Other lr arguments should be changed.',
    )

    p.add_argument(
        '--rl_lr',
        type=float,
        default=.01,
        help='Learning rate for reinforcement learning. Default=%(default)s'
    )
    p.add_argument(
        '--rl_n_samples',
        type=int,
        default=1,
        help='Number of samples to get baseline. Default=%(default)s'
    )

    p.add_argument(
        '--rl_init_epoch',
        type=int,
        default=0,
        help='Set initial epoch number, which can be useful in continue training. Default=%(default)s'
    )

    p.add_argument(
        '--rl_n_epochs',
        type=int,
        default=10,
        help='Number of epochs for reinforcement learning. Default=%(default)s'
    )
    p.add_argument(
        '--rl_n_gram',
        type=int,
        default=6,
        help='Maximum number of tokens to calculate BLEU for reinforcement learning. Default=%(default)s'
    )
    p.add_argument(
        '--rl_reward',
        type=str,
        default='gleu',
        help='Method name to use as reward function for RL training. Default=%(default)s'
    )

    p.add_argument(
        '--rl_batch_ratio',
        type=int,
        default=1,
        help='rl_batch_size = batch_size * rl_batch_ratio Default=%(default)s'
    )

    p.add_argument(
        '--use_transformer',
        action='store_true',
        help='Set model architecture as Transformer.',
    )
    p.add_argument(
        '--n_splits',
        type=int,
        default=8,
        help='Number of heads in multi-head attention in Transformer. Default=%(default)s',
    )

    config = p.parse_args()

    if (args is not None) and (not is_continue) :
        config.model_fn = args.model_filepath
        config.train = args.train_filepath
        config.valid = args.valid_filepath
        config.use_transformer = args.use_transformer
        config.n_splits = args.n_splits
        config.lang = args.language
        config.gpu_id = args.gpu_id
        config.batch_size = args.batch_size
        config.n_epochs = args.epochs
        config.verbose = args.verbose
        config.max_length = args.max_length
        config.dropout = args.dropout
        config.lr = args.lr
        config.lr_step = args.lr_step

        config.rl_lr = args.rl_lr
        config.rl_n_samples = args.rl_n_samples
        config.rl_init_epoch = args.rl_init_epoch
        config.rl_n_epochs = args.rl_n_epochs
        config.rl_n_gram = args.rl_n_gram
        config.rl_reward = args.rl_reward
        config.rl_batch_ratio = args.rl_batch_ratio

        
        config.max_grad_norm = args.max_grad_norm
        config.word_vec_size = args.word_vec_size
        config.hidden_size = args.hidden_size
        config.n_layers = args.layer_number
        config.use_adam = args.use_adam
        config.iteration_per_update = args.iteration_per_update
    return config

def get_model(input_size, output_size, config):
    if config.use_transformer:
        model = Transformer(
            input_size,                     # Source vocabulary size
            config.hidden_size,             # Transformer doesn't need word_vec_size.
            output_size,                    # Target vocabulary size
            n_splits=config.n_splits,       # Number of head in Multi-head Attention.
            n_enc_blocks=config.n_layers,   # Number of encoder blocks
            n_dec_blocks=config.n_layers,   # Number of decoder blocks
            dropout_p=config.dropout,       # Dropout rate on each block
        )
    else:
        model = Seq2Seq(
            input_size,
            config.word_vec_size,           # Word embedding vector size
            config.hidden_size,             # LSTM's hidden vector size
            output_size,
            n_layers=config.n_layers,       # number of layers in LSTM
            dropout_p=config.dropout        # dropout-rate in LSTM
        )

    return model


def get_crit(output_size, pad_index):
    # Default weight for loss equals to 1, but we don't need to get loss for PAD token.
    # Thus, set a weight for PAD to zero.
    loss_weight = torch.ones(output_size)
    loss_weight[pad_index] = 0.
    # Instead of using Cross-Entropy loss,
    # we can use Negative Log-Likelihood(NLL) loss with log-probability.
    crit = nn.NLLLoss(
        weight=loss_weight,
        reduction='sum'
    )

    return crit


def get_optimizer(model, config):
    if config.use_adam:
        if config.use_transformer:
            optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .98))
        else: # case of rnn based seq2seq.
            optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

    return optimizer


def get_scheduler(optimizer, config):
    if config.lr_step > 0:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[i for i in range(
                max(0, config.lr_decay_start - 1),
                config.n_epochs,
                config.lr_step
            )],
            gamma=config.lr_gamma,
            last_epoch=config.init_epoch if config.init_epoch > 0 else -1,
        )
    else:
        lr_scheduler = None

    return lr_scheduler


def main(config, model_weight=None, opt_weight=None, src_vocab = None, tgt_vocab = None):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    loader = DataLoader(
        config.train,                           # Train file name except extention, which is language.
        config.valid,                           # Validation file name except extension.
        (config.lang[:2], config.lang[-2:]),    # Source and target language.
        batch_size=config.batch_size,
        device=-1,                              # Lazy loading
        max_length=config.max_length,           # Loger sequence will be excluded.
        dsl=False,                              # Turn-off Dual-supervised Learning mode.
    )

    if src_vocab is not None and tgt_vocab is not None:
        loader.load_vocab(src_vocab, tgt_vocab)

    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
    model = get_model(input_size, output_size, config)
    crit = get_crit(output_size, data_loader.PAD)

    if model_weight is not None:
        model.load_state_dict(model_weight)

    # Pass models to GPU device if it is necessary.
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    optimizer = get_optimizer(model, config)

    if opt_weight is not None and (config.use_adam or config.use_radam):
        optimizer.load_state_dict(opt_weight)

    lr_scheduler = get_scheduler(optimizer, config)

    if config.verbose >= 2:
        print(model)
        print(crit)
        print(optimizer)

    if config.n_epochs > config.init_epoch:
        trainer = BaseTrainer(model, crit, optimizer
        ,train_loader=loader.train_iter
        ,valid_loader=loader.valid_iter
        ,lr_scheduler= lr_scheduler
        ,config=config
        ,src_vocab=loader.src.vocab
        ,tgt_vocab=loader.tgt.vocab)
        trainer.train()

    if config.rl_n_epochs > config.rl_init_epoch:
        batch_size = int(config.batch_size * config.rl_batch_ratio)

        loader.train_iter.batch_size = batch_size
        loader.valid_iter.batch_size = batch_size

        optimizer = optim.SGD(model.parameters(), lr=config.rl_lr)
        mrt_trainer = MinimumRiskTrainer(model, crit, optimizer
        ,train_loader=loader.train_iter
        ,valid_loader=loader.valid_iter
        ,lr_scheduler= lr_scheduler
        ,config=config
        ,src_vocab=loader.src.vocab
        ,tgt_vocab=loader.tgt.vocab)
        mrt_trainer.train()


    """"
    # Start training. This function maybe equivalant to 'fit' function in Keras.
    mle_trainer = SingleTrainer(MaximumLikelihoodEstimationEngine, config)
    mle_trainer.train(
        model,
        crit,
        optimizer,
        train_loader=loader.train_iter,
        valid_loader=loader.valid_iter,
        src_vocab=loader.src.vocab,
        tgt_vocab=loader.tgt.vocab,
        n_epochs=config.n_epochs,
        lr_scheduler=lr_scheduler,
    )
    """

if __name__ == '__main__':

    # args = NMTArgumets(model_filepath='NMT_2021_0915.pth'
    #                 , train_filepath='./corpus/corpus.shuf.train.tok.bpe.tr'
    #                 , valid_filepath='./corpus/corpus.shuf.valid.tok.bpe.tr'
    #                 , lr=1e-2
    #                 , max_grad_norm=1e+8
    #                 , batch_size=128
    #                 , epochs=5
    #                 , iteration_per_update=2
    #                 , hidden_size=256
    #                 , word_vec_size=256)


    
    # args = NMTArgumets(model_filepath='./2021.0919.NMT/NMT.pth'
    #                 , train_filepath='./corpus/corpus.shuf.train.tok.bpe.tr'
    #                 , valid_filepath='./corpus/corpus.shuf.valid.tok.bpe.tr'
    #                 , lr=1e-3
    #                 , max_grad_norm=1e+8
    #                 , batch_size=128
    #                 , epochs=30
    #                 , iteration_per_update=2
    #                 , hidden_size=512
    #                 , word_vec_size=768
    #                 , max_length=128
    #                 , dropout=.2
    #                 , is_shutdonw=True)

    cur_dir = os.path.dirname(__file__)
    if len(cur_dir) == 0 : cur_dir = '.'

    args = NMTArgumets(model_filepath=cur_dir+'/2021.1004.TFM/Transformer.pth'
                    , train_filepath=cur_dir+'/corpus/corpus.shuf.train.tok.bpe.tr'
                    , valid_filepath=cur_dir+'/corpus/corpus.shuf.valid.tok.bpe.tr'
                    , lr=1e-3
                    , lr_step=0
                    , rl_n_epochs = 0
                    , max_grad_norm=1e+8
                    , batch_size=128
                    , epochs=4
                    , iteration_per_update=32
                    , hidden_size=768
                    , word_vec_size=512
                    , max_length=64
                    , dropout=.2
                    , use_transformer=True
                    , is_shutdonw=False)
    
    config = define_argparser(is_continue=False, args = args)
    main(config)
    if args.is_shutdon:os.system('shutdown -s -f')
