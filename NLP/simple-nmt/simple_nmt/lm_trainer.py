from copy import deepcopy
from typing import Tuple

import numpy as np
import torch

from torch import optim
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from ignite.engine import Engine, Events
from torch.utils import data

from simple_nmt.trainer import MaximumLikelihoodEstimationEngine, BaseTrainer, TrainerSaveInterface

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

import simple_nmt.data_loader as data_loader

from Utilitis.utilitis import get_grad_norm, get_parameter_norm, printProgress, eos_insert, bos_insert, bos_remove
from Arguments import TrainerArguments


class LanguageModelTrainingEngine(MaximumLikelihoodEstimationEngine):

    def __init__(
        self,
        func,
        model,
        crit,
        optimizer,
        lr_scheduler,
        is_src_target,
        config
    ):
        self.is_src_target = is_src_target

        super().__init__(func, model, crit, optimizer, lr_scheduler, config)

        self.best_model = None
        self.scaler = GradScaler()

    @staticmethod
    def _eos_cutting(tgt):
        for i in range(len(tgt[0])):
            length = tgt[1][i]
            tgt[0][i, :length] = data_loader.PAD
        return tgt


    @staticmethod
    def train(engine, mini_batch):
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train()        
        engine.optimizer.zero_grad()

        device = next(engine.model.parameters()).device
        mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
        mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

        # if 'is_src_target' is true, the trainer would train language model for source language.
        # For dsl case, both x and y has BOS and EOS tokens.
        # Thus, we need to remove BOS and EOS before the training.
        x = mini_batch.src[0][:, :-1] if engine.is_src_target else mini_batch.tgt[0][:, :-1]
        y = mini_batch.src[0][:, 1:] if engine.is_src_target else mini_batch.tgt[0][:, 1:]
        # |x| = |y| = (batch_size, length)

        with autocast(not engine.config.off_autocast):
            y_hat = engine.model(x)
            # |y_hat| = (batch_size, length, output_size)

            loss = engine.crit(
                y_hat.contiguous().view(-1, y_hat.size(-1)),
                y.contiguous().view(-1),
            ).sum()
            backward_target = loss.div(y.size(0))

        if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
            engine.scaler.scale(backward_target).backward()
        else:
            backward_target.backward()

        word_count = int(mini_batch.src[1].sum()) if engine.is_src_target else int(mini_batch.tgt[1].sum())
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        # In orther to avoid gradient exploding, we apply gradient clipping.
        torch_utils.clip_grad_norm_(
            engine.model.parameters(),
            engine.config.max_grad_norm,
        )
        # Take a step of gradient descent.
        if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
            # Use scaler instead of engine.optimizer.step() if using GPU.
            engine.scaler.step(engine.optimizer)
            engine.scaler.update()
        else:
            engine.optimizer.step()

        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl,
            '|param|': p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
            '|g_param|': g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            device = next(engine.model.parameters()).device
            mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
            mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

            x = mini_batch.src[0][:, :-1] if engine.is_src_target else mini_batch.tgt[0][:, :-1]
            y = mini_batch.src[0][:, 1:] if engine.is_src_target else mini_batch.tgt[0][:, 1:]
            # |x| = |y| = (batch_size, length)

            with autocast(not engine.config.off_autocast):
                y_hat = engine.model(x)
                # |y_hat| = (batch_size, length, output_size)

                loss = engine.crit(y_hat.contiguous().view(-1, y_hat.size(-1)),
                                   y.contiguous().view(-1),
                                   ).sum()
            
        word_count = int(mini_batch.src[1].sum()) if engine.is_src_target else int(mini_batch.tgt[1].sum())
        loss = float(loss / word_count)
        ppl = np.exp(loss)
        
        return {
            'loss': loss,
            'ppl': ppl,
        }

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        pass


class LanguageModelTrainer():

    def __init__(self, config:TrainerArguments):
        self.config = config

    def train(
        self,
        model,
        crit, optimizer,
        train_loader, valid_loader,
        src_vocab, tgt_vocab,
        n_epochs,
        lr_scheduler=None
    ):
        if src_vocab is not None and tgt_vocab is not None:
            raise NotImplementedError('You should assign None one of vocab to designate target language.')
        if src_vocab is None:
            is_src_target = False
        elif tgt_vocab is None:
            is_src_target = True
        else:
            raise NotImplementedError('You cannot assign None both vocab.')

        trainer = LanguageModelTrainingEngine(
            LanguageModelTrainingEngine.train,
            model,
            crit,
            optimizer,
            lr_scheduler,
            is_src_target,
            self.config,
        )
        evaluator = LanguageModelTrainingEngine(
            LanguageModelTrainingEngine.validate,
            model,
            crit,
            optimizer=None,
            lr_scheduler=None,
            is_src_target=is_src_target,
            config=self.config,
        )

        LanguageModelTrainingEngine.attach(trainer, evaluator, verbose=self.config.verbose)

        def run_validation(engine, evaluator, valid_loader):
            evaluator.run(valid_loader, max_epochs=1)

            if engine.lr_scheduler is not None:
                engine.lr_scheduler.step()

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, run_validation, evaluator, valid_loader
        )
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, LanguageModelTrainingEngine.check_best
        )
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            LanguageModelTrainingEngine.save_model,
            trainer,
            self.config,
            src_vocab,
            tgt_vocab,
        )

        trainer.run(train_loader, max_epochs=n_epochs)

        if n_epochs > 0:
            model.load_state_dict(evaluator.best_model)

        return model


class Language_Model_Trainer(BaseTrainer):
    def __init__(self
        ,model
        ,crit
        ,optimizer
        ,train_loader
        ,valid_loader
        ,config:TrainerArguments
        ,src_vocab
        ,tgt_vocab        
        ,save_interface:TrainerSaveInterface
        ,lr_scheduler=None
        ,save_keyword:str = ''
    ):
        super().__init__(model
        ,crit
        ,optimizer
        ,train_loader
        ,valid_loader
        ,lr_scheduler
        ,config
        ,src_vocab
        ,tgt_vocab
        ,save_interface
        )
        self.is_src_target = False
        if src_vocab is not None and tgt_vocab is not None:
            raise NotImplementedError('You should assign None one of vocab to designate target language.')
        if src_vocab is None:
            self.is_src_target = False
        elif tgt_vocab is None:
            self.is_src_target = True
        else:
            raise NotImplementedError('You cannot assign None both vocab.')

        self.save_keyword = save_keyword


    def do_train(self, max_iteration)->dict:
        for index, mini_batch in enumerate(self.train_loader):
            self.iteration += 1
            # You have to reset the gradients of all model parameters
            # before to take another step in gradient descent.
            self.model.train()        
            self.optimizer.zero_grad()

            mini_batch.src = (mini_batch.src[0].to(self.device), mini_batch.src[1])
            mini_batch.tgt = (mini_batch.tgt[0].to(self.device), mini_batch.tgt[1])

            # if 'is_src_target' is true, the trainer would train language model for source language.
            # For dsl case, both x and y has BOS and EOS tokens.
            # Thus, we need to remove BOS and EOS before the training.
            src = mini_batch.src if self.is_src_target else mini_batch.tgt
            x = src[0]
            y = bos_remove(src, data_loader.PAD)[0]

            # x = mini_batch.src[0] if self.is_src_target else mini_batch.tgt[0][:, :-1]
            # y = mini_batch.src[0] if self.is_src_target else mini_batch.tgt[0][:, 1:]
            # |x| = |y| = (batch_size, length)

            with autocast(self.config.use_autocast):
                y_hat = self.model(x)
                # |y_hat| = (batch_size, length, output_size)

                loss = self.crit(
                    y_hat.contiguous().view(-1, y_hat.size(-1)),
                    y.contiguous().view(-1),
                ).sum()
                backward_target = loss.div(y.size(0))

            if self.config.gpu_id >= 0 and self.config.use_autocast:
                self.scaler.scale(backward_target).backward()
            else:
                backward_target.backward()

            word_count = int(mini_batch.src[1].sum()) if self.is_src_target else int(mini_batch.tgt[1].sum())
            p_norm = float(get_parameter_norm(self.model.parameters()))
            g_norm = float(get_grad_norm(self.model.parameters()))

            # In orther to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            # Take a step of gradient descent.
            if self.config.gpu_id >= 0 and self.config.use_autocast:
                # Use scaler instead of engine.optimizer.step() if using GPU.
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            loss = float(loss / word_count)
            ppl = np.exp(loss)

            fix = 'loss : {:.2f}  ppl : {:.2f}  |param| : {:.2f}  |g_param| : {:.2f}'.format(loss, ppl, g_norm, p_norm)
            printProgress(index, len(self.train_loader), prefix=fix)
        return {'loss':loss , 'ppl':ppl ,'param':p_norm,'g_param':g_norm}
            

    def do_valid(self)->dict:
        for index, mini_batch in enumerate(self.valid_loader):
            self.model.eval()

            with torch.no_grad():
                device = next(self.model.parameters()).device
                mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
                mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

                src = mini_batch.src if self.is_src_target else mini_batch.tgt
                x = src[0]
                y = bos_remove(src, data_loader.PAD)[0]

                # x = mini_batch.src[0] if self.is_src_target else mini_batch.tgt[0][:, :-1]
                # y = mini_batch.src[0] if self.is_src_target else mini_batch.tgt[0][:, 1:]
                # |x| = |y| = (batch_size, length)

                with autocast(self.config.use_autocast):
                    y_hat = self.model(x)
                    # |y_hat| = (batch_size, length, output_size)

                    loss = self.crit(y_hat.contiguous().view(-1, y_hat.size(-1)),
                                    y.contiguous().view(-1),
                                    ).sum()
            
            word_count = int(mini_batch.src[1].sum()) if self.is_src_target else int(mini_batch.tgt[1].sum())
            loss = float(loss / word_count)
            ppl = np.exp(loss)

            fix = 'loss : {:.5f}  ppl : {:.4f}'.format(loss, ppl)
            printProgress(index, len(self.valid_loader), prefix=fix)
        return {'loss':loss , 'ppl':ppl}

    def check_best(self, loss:float):
        loss = float(loss)
        if loss <= self.best_loss:
            self.best_loss = loss
            self.best_model = deepcopy(self.model.state_dict())





        




    
        
        