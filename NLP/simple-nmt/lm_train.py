import os
import pprint

import torch
from torch import optim

import sys
sys.path.append('D:\\work\\DeepLearning')

from Arguments import TrainerArguments,MODEL_FILE, OPTIMAIZER_ADAM, OPTIMAZIER_SGD


from simple_nmt.data_loader import DataLoader
import simple_nmt.data_loader as data_loader

from simple_nmt.models.rnnlm import LanguageModel
from simple_nmt.lm_trainer import Language_Model_Trainer as LMTrainer
from simple_nmt.trainer import BaseTrainer, TrainerSaveInterface



from dual_train import get_crits

class LMTSaveLoad(TrainerSaveInterface):
    def save(self, trainer:BaseTrainer, kwargs:dict = None):
        trainer:LMTrainer = trainer
        config = trainer.config
        save_folder = config.save_folder
        torch.save(trainer.best_model , save_folder+trainer.save_keyword+'.'+MODEL_FILE) 
        optim_file_path = save_folder + trainer.save_keyword+'.'+OPTIMAIZER_ADAM if config.use_adam else OPTIMAZIER_SGD
        torch.save(trainer.optimizer.state_dict(), optim_file_path)

    def train_complet_save(self, config:TrainerArguments, src_vocab, tgt_vocab):
        save_folder = config.save_folder
        torch.save(config, save_folder+'lmt_config')
        torch.save(src_vocab, save_folder+'lmt_src.vocab')
        torch.save(tgt_vocab, save_folder+'lmt_tgt.vocab')
        

def get_models(src_vocab_size, tgt_vocab_size, config:TrainerArguments):
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


def main(config:TrainerArguments):
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

    print(models)

    save_interface = LMTSaveLoad()

    modelcnt = 0
    for model, crit in zip(models, crits):
        optimizer = optim.Adam(model.parameters())
        modelcnt += 1

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        lm_trainer = LMTrainer(
            model, crit, optimizer,
            train_loader=loader.train_iter,
            valid_loader=loader.valid_iter,
            config=config,
            save_interface=save_interface,
            src_vocab=loader.src.vocab if model.vocab_size == src_vocab_size else None,
            tgt_vocab=loader.tgt.vocab if model.vocab_size == tgt_vocab_size else None,
            save_keyword='lmt{}'.format(modelcnt)
            )
        lm_trainer.train()
        config.init_epoch = 0
        if config.gpu_id >= 0:
            model.cpu()
            crit.cpu()

    save_interface.train_complet_save(config, loader.src.vocab, loader.tgt.vocab)
    
    """
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
    """

if __name__ == '__main__':

    cur_dir = os.path.dirname(__file__)
    if len(cur_dir) == 0 : cur_dir = '.'

    """
    config = TrainerArguments(save_folder=cur_dir+'/2021.1029.DSL'
                    ,train_filepath=cur_dir+'/corpus/corpus.shuf.train.tok.bpe.tr'
                    ,valid_filepath=cur_dir+'/corpus/corpus.shuf.valid.tok.bpe.tr'
                    ,is_use_adam=True
                    ,gpu_id=0
                    ,epochs=30
                    ,dropout=0.2
                    ,max_grad_norm=1e+8
                    )
    """
    config = TrainerArguments(save_folder=cur_dir+'/2021.1029.DSL'
                    ,train_filepath=cur_dir+'/corpus/corpus.shuf.train.tok.bpe.tr'
                    ,valid_filepath=cur_dir+'/corpus/corpus.shuf.valid.tok.bpe.tr'
                    # ,train_filepath=cur_dir+'/corpus/1500_train_corpus.tr'
                    # ,valid_filepath=cur_dir+'/corpus/1500_valid_corpus.tr'
                    ,use_adam=True
                    ,gpu_id=0
                    ,batch_size=192
                    ,epochs=2
                    ,dropout=0.2
                    ,max_grad_norm=1e+8
                    )
    
    main(config)
