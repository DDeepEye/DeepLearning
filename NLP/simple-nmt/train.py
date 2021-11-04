from re import I
import sys
sys.path.append('D:\\work\\DeepLearning')
import os
import argparse
import pprint

import torch
from torch import optim
import torch.nn as nn
from torch.serialization import save

import torch_optimizer as custom_optim

from simple_nmt.data_loader import DataLoader
import simple_nmt.data_loader as data_loader

from simple_nmt.models.seq2seq import Seq2Seq
from simple_nmt.models.transformer import Transformer

from simple_nmt.trainer import BaseTrainer,TrainerSaveInterface
from simple_nmt.rl_trainer import MinimumRiskTrainer

import Arguments as Args
from Arguments import TrainerArguments

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Arguments import NMTArgumets


class TrainerLoadInterface:
    def load(self, trainer:BaseTrainer, **kwargs):   
        save_folder = kwargs['save_folder']
        model = torch.load(save_folder + Args.MODEL_FILE)
        trainer.model.load_state_dict(model)
        optim_file_path = save_folder + Args.OPTIMAIZER_ADAM if config.use_adam else Args.OPTIMAZIER_SGD
        optimizer = torch.load(optim_file_path)
        trainer.optimizer.load_state_dict(optimizer)
        trainer.config = torch.load(save_folder+'config')
        data_loader : DataLoader = kwargs['data_loader']
        src_vocab = torch.load(config.save_folder+'src.vocab')
        tgt_vocab = torch.load(config.save_folder+'tgt.vocab')
        data_loader.load_vocab(src_vocab, tgt_vocab)

class BaseSaveLoad(TrainerSaveInterface, TrainerLoadInterface):
    def save(self, trainer:BaseTrainer, **kwargs):
        config = trainer.config
        torch.save(trainer.model.state_dict(), config.save_folder+Args.MODEL_FILE) 
        optim_file_path = config.save_folder + Args.OPTIMAIZER_ADAM if config.use_adam else Args.OPTIMAZIER_SGD
        torch.save(trainer.optimizer.state_dict(), optim_file_path)
        torch.save(config, config.save_folder + 'config')
        torch.save(trainer.src_vocab, config.save_folder+'src.vocab')
        torch.save(trainer.tgt_vocab, config.save_folder+'tgt.vocab')

        super().save(trainer, kwargs)

        with open(trainer.config.save_folder+'log.txt', 'wt', encoding='utf-8') as file:
            file.writelines('epoch : {}\n'.format(trainer.config.init_epoch))
            file.writelines('n_layers : {}\n'.format(trainer.config.layer_number))
            file.writelines('max_length : {}\n'.format(trainer.config.max_length))
            file.writelines('batch_size : {}\n'.format(trainer.config.batch_size))
            file.writelines('hidden size : {}\n'.format(trainer.config.hidden_size))
            file.writelines('word_vec_size : {}\n'.format(trainer.config.word_vec_size))
            file.writelines('iteration_per_update : {}\n'.format(trainer.config.iteration_per_update))
            file.writelines('use_adam : {}\n'.format(trainer.config.use_adam))
    


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


def main(config:TrainerArguments, model_weight=None, opt_weight=None, src_vocab = None, tgt_vocab = None):
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
        ,tgt_vocab=loader.tgt.vocab
        ,save_interface=BaseSaveLoad())
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
        ,tgt_vocab=loader.tgt.vocab
        ,save_interface=BaseSaveLoad())
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


    
    

    cur_dir = os.path.dirname(__file__)
    if len(cur_dir) == 0 : cur_dir = '.'

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

    # config = NMTArgumets(save_folder=cur_dir+'/2021.11.03.x2y/'
    #                 , train_filepath=cur_dir+'/corpus/corpus.shuf.train.tok.bpe.tr'
    #                 , valid_filepath=cur_dir+'/corpus/corpus.shuf.valid.tok.bpe.tr'
    #                 , lr=1e-3
    #                 , lr_step=0
    #                 , rl_n_epochs = 0
    #                 , max_grad_norm=1e+8
    #                 , batch_size=128
    #                 , epochs=1
    #                 , iteration_per_update=32
    #                 , hidden_size=768
    #                 , word_vec_size=512
    #                 , max_length=64
    #                 , dropout=.2
    #                 , use_transformer=False
    #                 , is_shutdown=False)

    config = NMTArgumets(save_folder=cur_dir+'/2021.11.03.x2y/'
                    , train_filepath=cur_dir+'/corpus/corpus.shuf.train.tok.bpe.tr'
                    , valid_filepath=cur_dir+'/corpus/corpus.shuf.valid.tok.bpe.tr'
                    , lr=1e-3
                    , lr_step=0
                    , max_grad_norm=1e+8
                    , batch_size=128
                    , epochs=1
                    , iteration_per_update=32
                    , hidden_size=768
                    , word_vec_size=512
                    , max_length=128
                    , dropout=.2
                    , use_transformer=False
                    , is_shutdown=False)
    
    main(config)
    if config.is_shutdown:os.system('shutdown -s -f')
