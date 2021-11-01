import os
import argparse
import pprint

import torch
from torch import optim
import torch.nn as nn

import sys
sys.path.append('D:\\work\\DeepLearning')

from simple_nmt.data_loader import DataLoader
import simple_nmt.data_loader as data_loader

from simple_nmt.models.seq2seq import Seq2Seq
from simple_nmt.models.transformer import Transformer
from simple_nmt.models.rnnlm import LanguageModel

from simple_nmt.lm_trainer import LanguageModelTrainer as LMTrainer
from simple_nmt.dual_trainer import Dual_Supervised_Trainer as DSLTrainer , DualTrainerSaveInterface
from Arguments import DualTrainerArgs,MODEL_FILE, OPTIMAIZER_ADAM, OPTIMAZIER_SGD


def load_lm(save_folder, language_models):
    model1 = torch.load(save_folder+'lmt1.'+MODEL_FILE, map_location='cpu')
    model2 = torch.load(save_folder+'lmt2.'+MODEL_FILE, map_location='cpu')

    language_models[0].load_state_dict(model1)
    language_models[1].load_state_dict(model2)


def get_models(src_vocab_size, tgt_vocab_size, config:DualTrainerArgs):
    language_models = [
        LanguageModel(
            tgt_vocab_size,
            config.word_vec_size,
            config.hidden_size,
            n_layers=config.layer_number,
            dropout_p=config.dropout,
        ),
        LanguageModel(
            src_vocab_size,
            config.word_vec_size,
            config.hidden_size,
            n_layers=config.layer_number,
            dropout_p=config.dropout,
        ),
    ]

    if config.use_transformer:
        models = [
            Transformer(
                src_vocab_size,
                config.hidden_size,
                tgt_vocab_size,
                n_splits=config.n_splits,
                n_enc_blocks=config.layer_number,
                n_dec_blocks=config.layer_number,
                dropout_p=config.dropout,
            ),
            Transformer(
                tgt_vocab_size,
                config.hidden_size,
                src_vocab_size,
                n_splits=config.n_splits,
                n_enc_blocks=config.layer_number,
                n_dec_blocks=config.layer_number,
                dropout_p=config.dropout,
            ),
        ]
    else:
        models = [
            Seq2Seq(
                src_vocab_size,
                config.word_vec_size,
                config.hidden_size,
                tgt_vocab_size,
                n_layers=config.layer_number,
                dropout_p=config.dropout,
            ),
            Seq2Seq(
                tgt_vocab_size,
                config.word_vec_size,
                config.hidden_size,
                src_vocab_size,
                n_layers=config.layer_number,
                dropout_p=config.dropout,
            ),
        ]

    return language_models, models


def get_crits(src_vocab_size, tgt_vocab_size, pad_index):
    loss_weights = [
        torch.ones(tgt_vocab_size),
        torch.ones(src_vocab_size),
    ]
    loss_weights[0][pad_index] = .0
    loss_weights[1][pad_index] = .0

    crits = [
        nn.NLLLoss(weight=loss_weights[0], reduction='none'),
        nn.NLLLoss(weight=loss_weights[1], reduction='none'),
    ]

    return crits


def get_optimizers(models, config):
    if config.use_transformer:
        optimizers = [
            optim.Adam(models[0].parameters(), betas=(.9, .98)),
            optim.Adam(models[1].parameters(), betas=(.9, .98)),
        ]
    else:
        optimizers = [
            optim.Adam(models[0].parameters()),
            optim.Adam(models[1].parameters()),
        ]

    return optimizers


def main(config:DualTrainerArgs, model_weight=None, opt_weight=None):

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(config))

    loader = DataLoader(
        config.train_filepath,
        config.valid_filepath,
        (config.language[:2], config.language[-2:]),
        batch_size=config.batch_size,
        device=-1,
        max_length=config.max_length,
        dsl=True,
    )

    src_vocab = torch.load(config.save_folder+'lmt_src.vocab')
    tgt_vocab = torch.load(config.save_folder+'lmt_tgt.vocab')

    loader.load_vocab(src_vocab, tgt_vocab)

    src_vocab_size = len(loader.src.vocab)
    tgt_vocab_size = len(loader.tgt.vocab)

    language_models, models = get_models(
        src_vocab_size,
        tgt_vocab_size,
        config
    )

    crits = get_crits(
        src_vocab_size,
        tgt_vocab_size,
        pad_index=data_loader.PAD
    )

    if model_weight is not None:
        for model, w in zip(models + language_models, model_weight):
            model.load_state_dict(w)

    load_lm(config.save_folder, language_models)

    if config.gpu_id >= 0:
        for lm, seq2seq, crit in zip(language_models, models, crits):
            lm.cuda(config.gpu_id)
            seq2seq.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

    optimizers = get_optimizers(models, config)

    if opt_weight is not None:
        for opt, w in zip(optimizers, opt_weight):
            opt.load_state_dict(w)

    print(language_models)
    print(models)
    print(crits)
    print(optimizers)

    dsl_trainer = DSLTrainer(config
        ,models
        ,language_models
        ,crits
        ,optimizers
        ,loader.train_iter
        ,loader.valid_iter
        ,vocabs=[loader.src.vocab, loader.tgt.vocab]
        ,lr_schedulers=None
        )

    dsl_trainer.train()


if __name__ == '__main__':

    cur_dir = os.path.dirname(__file__)
    if len(cur_dir) == 0 : cur_dir = '.'

    config = DualTrainerArgs(save_folder=cur_dir+'/2021.1029.DSL'
                    # ,train_filepath=cur_dir+'/corpus/corpus.shuf.train.tok.bpe.tr'
                    # ,valid_filepath=cur_dir+'/corpus/corpus.shuf.valid.tok.bpe.tr'
                    ,train_filepath=cur_dir+'/corpus/1500_train_corpus.tr'
                    ,valid_filepath=cur_dir+'/corpus/1500_valid_corpus.tr'
                    ,use_adam=True
                    ,gpu_id=0
                    ,batch_size=64
                    ,epochs=1
                    ,dropout=0.2
                    ,max_grad_norm=1
                    ,dsl_n_warmup_epochs = 1
                    ,dsl_lambda = 1e-2
                    ,is_shutdown=False
                    )

    main(config)
