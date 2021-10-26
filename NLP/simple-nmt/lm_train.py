import os
import argparse
import pprint

import torch
from torch import optim
import torch.nn as nn

from simple_nmt.data_loader import DataLoader
import simple_nmt.data_loader as data_loader

from simple_nmt.models.rnnlm import LanguageModel
from simple_nmt.lm_trainer import LanguageModelTrainer as LMTrainer
from simple_nmt.trainer import BaseTrainer

from DeepLearning.Arguments import Arguments
from dual_train import get_crits

def get_models(src_vocab_size, tgt_vocab_size, config:Arguments):
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

    return language_models


def main(config:Arguments):
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

    src_vocab_size = len(loader.src.vocab)
    tgt_vocab_size = len(loader.tgt.vocab)

    models = get_models(
        src_vocab_size,
        tgt_vocab_size,
        config
    )

    crits = get_crits(
        src_vocab_size,
        tgt_vocab_size,
        pad_index=data_loader.PAD
    )

    if config.gpu_id >= 0:
        for model, crit in zip(models, crits):
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

    print(models)

    for model, crit in zip(models, crits):
        optimizer = optim.Adam(model.parameters())
        lm_trainer = LMTrainer(config)

        model = lm_trainer.train(
            model, crit, optimizer,
            train_loader=loader.train_iter,
            valid_loader=loader.valid_iter,
            src_vocab=loader.src.vocab if model.vocab_size == src_vocab_size else None,
            tgt_vocab=loader.tgt.vocab if model.vocab_size == tgt_vocab_size else None,
            n_epochs=config.n_epochs,
        )

    torch.save(
        {
            'model': [
                models[0].state_dict(),
                models[1].state_dict(),
            ],
            'config': config,
            'src_vocab': loader.src.vocab,
            'tgt_vocab': loader.tgt.vocab,
        }, config.model_fn
    )

if __name__ == '__main__':

    cur_dir = os.path.dirname(__file__)
    if len(cur_dir) == 0 : cur_dir = '.'

    config = Arguments(save_folder=cur_dir+'/2021.1026.DSL'
                    ,train_filepath=cur_dir+'/corpus/corpus.shuf.train.tok.bpe.tr'
                    ,valid_filepath=cur_dir+'/corpus/corpus.shuf.valid.tok.bpe.tr'
                    ,is_use_adam=True
                    ,gpu_id=0
                    ,epochs=30
                    ,dropout=0.2
                    ,max_grad_norm=1e+8
                    )
    
    main(config)
