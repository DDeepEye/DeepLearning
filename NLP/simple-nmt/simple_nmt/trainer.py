import sys
import os
import numpy as np

import torch
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from Utilitis.utilitis import get_grad_norm, get_parameter_norm, printProgress
from Arguments import TrainerArguments


VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MaximumLikelihoodEstimationEngine(Engine):

    def __init__(self, func, model, crit, optimizer, lr_scheduler, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config

        super().__init__(func)

        self.best_loss = np.inf
        self.scaler = GradScaler()

    @staticmethod
    #@profile
    def train(engine, mini_batch):
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train()
        if engine.state.iteration % engine.config.iteration_per_update == 1 or \
            engine.config.iteration_per_update == 1:
            if engine.state.iteration > 1:
                engine.optimizer.zero_grad()

        device = next(engine.model.parameters()).device
        mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
        mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

        # Raw target variable has both BOS and EOS token. 
        # The output of sequence-to-sequence does not have BOS token. 
        # Thus, remove BOS token for reference.
        x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
        # |x| = (batch_size, length)
        # |y| = (batch_size, length)

        with autocast(not engine.config.off_autocast):
            # Take feed-forward
            # Similar as before, the input of decoder does not have EOS token.
            # Thus, remove EOS token for decoder input.
            y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
            # |y_hat| = (batch_size, length, output_size)

            loss = engine.crit(
                y_hat.contiguous().view(-1, y_hat.size(-1)),
                y.contiguous().view(-1)
            )
            backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)

        if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
            engine.scaler.scale(backward_target).backward()
        else:
            backward_target.backward()

        word_count = int(mini_batch.tgt[1].sum())
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        if engine.state.iteration % engine.config.iteration_per_update == 0 and \
            engine.state.iteration > 0:
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

            x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)

            with autocast(not engine.config.off_autocast):
                y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
                # |y_hat| = (batch_size, n_classes)
                loss = engine.crit(
                    y_hat.contiguous().view(-1, y_hat.size(-1)),
                    y.contiguous().view(-1),
                )
        
        word_count = int(mini_batch.tgt[1].sum())
        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl,
        }

    @staticmethod
    def attach(
        train_engine, validation_engine,
        training_metric_names = ['loss', 'ppl', '|param|', '|g_param|'],
        validation_metric_names = ['loss', 'ppl'],
        verbose=VERBOSE_BATCH_WISE,
    ):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics['|param|']
                avg_g_norm = engine.state.metrics['|g_param|']
                avg_loss = engine.state.metrics['loss']

                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} ppl={:.2f}'.format(
                    engine.state.epoch,
                    avg_p_norm,
                    avg_g_norm,
                    avg_loss,
                    np.exp(avg_loss),
                ))

        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_loss = engine.state.metrics['loss']

                print('Validation - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}'.format(
                    avg_loss,
                    np.exp(avg_loss),
                    engine.best_loss,
                    np.exp(engine.best_loss),
                ))

    @staticmethod
    def resume_training(engine, resume_epoch):
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = (resume_epoch - 1)

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        avg_train_loss = train_engine.state.metrics['loss']
        avg_valid_loss = engine.state.metrics['loss']

        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model_fn.split('.')
        
        model_fn = model_fn[:-1] + ['%02d' % train_engine.state.epoch,
                                    '%.2f-%.2f' % (avg_train_loss,
                                                   np.exp(avg_train_loss)
                                                   ),
                                    '%.2f-%.2f' % (avg_valid_loss,
                                                   np.exp(avg_valid_loss)
                                                   )
                                    ] + [model_fn[-1]]

        model_fn = '.'.join(model_fn)

        # Unlike other tasks, we need to save current model, not best model.
        torch.save(
            {
                'model': engine.model.state_dict(),
                'opt': train_engine.optimizer.state_dict(),
                'config': config,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
            }, model_fn
        )
    

class SingleTrainer():

    def __init__(self, target_engine_class, config):
        self.target_engine_class = target_engine_class
        self.config = config

    def train(
        self,
        model, crit, optimizer,
        train_loader, valid_loader,
        src_vocab, tgt_vocab,
        n_epochs,
        lr_scheduler=None
    ):
        # Declare train and validation engine with necessary objects.
        train_engine = self.target_engine_class(
            self.target_engine_class.train,
            model,
            crit,
            optimizer,
            lr_scheduler,
            self.config
        )
        validation_engine = self.target_engine_class(
            self.target_engine_class.validate,
            model,
            crit,
            optimizer=None,
            lr_scheduler=None,
            config=self.config
        )

        # Do necessary attach procedure to train & validation engine.
        # Progress bar and metric would be attached.
        self.target_engine_class.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        # After every train epoch, run 1 validation epoch.
        # Also, apply LR scheduler if it is necessary.
        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

            if engine.lr_scheduler is not None:
                engine.lr_scheduler.step()

        # Attach above call-back function.
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            validation_engine,
            valid_loader
        )
        # Attach other call-back function for initiation of the training.
        train_engine.add_event_handler(
            Events.STARTED,
            self.target_engine_class.resume_training,
            self.config.init_epoch,
        )

        # Attach validation loss check procedure for every end of validation epoch.
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, self.target_engine_class.check_best
        )
        # Attach model save procedure for every end of validation epoch.
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            self.target_engine_class.save_model,
            train_engine,
            self.config,
            src_vocab,
            tgt_vocab,
        )

        # Start training.
        train_engine.run(train_loader, max_epochs=n_epochs)

        return model




class BaseTrainer():
    def __init__(self
        ,model
        ,crit
        ,optimizer
        ,train_loader
        ,valid_loader
        ,lr_scheduler
        ,config : TrainerArguments
        ,src_vocab
        ,tgt_vocab
        ,save_interface
        ):
        
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_scheduler = lr_scheduler
        self.config = config

        self.best_model = None
        self.best_loss = np.inf
        self.scaler = GradScaler()
        self.device = torch.device('cuda:{}'.format(config.gpu_id) if config.gpu_id > -1 else 'cpu')
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.iteration = 0
        self.save_interface:TrainerSaveInterface = save_interface

    """"
    def save_model(self, log:dict):
        config = self.config
        torch.save(self.model.state_dict() , config.save_folder+Args.MODEL_FILE) 
        optim_file_path = config.save_folder + Args.OPTIMAIZER_ADAM if config.use_adam else Args.OPTIMAZIER_SGD
        torch.save(self.optimizer.state_dict(), optim_file_path)
        torch.save(config, config.save_folder + 'config')
        torch.save(self.src_vocab, config.save_folder+'src.vocab')
        torch.save(self.tgt_vocab, config.save_folder+'tgt.vocab')

        with open(self.config.save_folder+'log.txt', 'wt', encoding='utf-8') as file:
            for key in log.keys():
                file.writelines('{} : {}\n'.format(key, log[key]))

            file.writelines('epoch : {}\n'.format(self.config.init_epoch))
            file.writelines('n_layers : {}\n'.format(self.config.layer_number))
            file.writelines('max_length : {}\n'.format(self.config.max_length))
            file.writelines('batch_size : {}\n'.format(self.config.batch_size))
            file.writelines('hidden size : {}\n'.format(self.config.hidden_size))
            file.writelines('word_vec_size : {}\n'.format(self.config.word_vec_size))
            file.writelines('iteration_per_update : {}\n'.format(self.config.iteration_per_update))
            file.writelines('use_adam : {}\n'.format(self.config.use_adam))
    """

    def do_train(self, max_iteration)->dict:
        for index, mini_batch in enumerate(self.train_loader):
            self.iteration += 1
            self.model.train()
            if (self.iteration > 1) and \
                ((self.iteration % self.config.iteration_per_update == 1) or (self.config.iteration_per_update == 1)):
                self.optimizer.zero_grad()

            mini_batch.src = (mini_batch.src[0].to(self.device), mini_batch.src[1])
            mini_batch.tgt = (mini_batch.tgt[0].to(self.device), mini_batch.tgt[1])

            # Raw target variable has both BOS and EOS token. 
            # The output of sequence-to-sequence does not have BOS token. 
            # Thus, remove BOS token for reference.
            x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)

            with autocast(self.config.use_autocast):
                # Take feed-forward
                # Similar as before, the input of decoder does not have EOS token.
                # Thus, remove EOS token for decoder input.
                y_hat = self.model(x, mini_batch.tgt[0][:, :-1])
                # |y_hat| = (batch_size, length, output_size)

                train_loss = self.crit(
                    y_hat.contiguous().view(-1, y_hat.size(-1)),
                    y.contiguous().view(-1)
                )
                backward_target = train_loss.div(y.size(0)).div(self.config.iteration_per_update)

            if self.config.gpu_id >= 0 and self.config.use_autocast:
                self.scaler.scale(backward_target).backward()
            else:
                backward_target.backward()

            word_count = int(mini_batch.tgt[1].sum())
            p_norm = float(get_parameter_norm(self.model.parameters()))
            g_norm = float(get_grad_norm(self.model.parameters()))

            if (self.iteration % self.config.iteration_per_update == 0 and self.iteration > 0) or \
                (self.iteration == max_iteration):
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

            train_loss = float(train_loss / word_count)
            ppl = np.exp(train_loss)

            fix = 'loss : {:.5f}  ppl : {:.4f}  |param| : {:.4f}  |g_param| : {:.4f}'.format(train_loss, ppl, p_norm, g_norm)
            printProgress(index, len(self.train_loader), prefix=fix)
        return {'loss':train_loss , 'ppl':ppl ,'param':p_norm,'g_param':g_norm}
            

    def do_valid(self)->dict:
        for index, mini_batch in enumerate(self.valid_loader):
            self.model.eval()

            with torch.no_grad():
                device = next(self.model.parameters()).device
                mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
                mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

                x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
                # |x| = (batch_size, length)
                # |y| = (batch_size, length)

                with autocast(self.config.use_autocast):
                    y_hat = self.model(x, mini_batch.tgt[0][:, :-1])
                    # |y_hat| = (batch_size, n_classes)
                    valid_loss = self.crit(
                        y_hat.contiguous().view(-1, y_hat.size(-1)),
                        y.contiguous().view(-1),
                    )
            
            word_count = int(mini_batch.tgt[1].sum())
            valid_loss = float(valid_loss / word_count)
            ppl = np.exp(valid_loss)

            fix = 'loss : {:.5f}  ppl : {:.4f}'.format(valid_loss, ppl)
            printProgress(index, len(self.valid_loader), prefix=fix)
        return {'loss':valid_loss , 'ppl':ppl}

    def _next_epoch(self):
        self.config.init_epoch += 1
        return self.config.init_epoch

    def _get_epochs(self):
        max_epochs = self.config.epochs - self.config.init_epoch
        max_iteration = max_epochs * len(self.train_loader)
        return self.config.init_epoch, self.config.epochs, max_epochs ,max_iteration
        
    def _warnning(self):
        print('warnning!!! 현재 epoch {} 까지 모든 학습이 끝났습니다. \n학습을 연장하고 싶으시면 epoch 설정을 다시 하세요'.format(self.config.epochs))

    def check_best(self, loss:float):
        pass

    def train(self):
        current_epoch, epochs, max_epochs, max_iteration = self._get_epochs()

        if epochs > current_epoch:
            for e in range(max_epochs):
                fix = '\n current epoch {} / max epoch {} \n'.format(current_epoch+1, epochs)
                printProgress(e, epochs, prefix=fix)

                self.do_train(max_iteration)
                log = self.do_valid()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                current_epoch = self._next_epoch()
                self.check_best(log['loss'])
                if self.save_interface is not None:
                    self.save_interface.save(self, {'log' : log})
        else:
            self._warnning()


class TrainerSaveInterface():
    def save(self, trainer:BaseTrainer, kwargs:dict = None):
        if kwargs is not None:
            if 'log' in kwargs.keys():
                log:dict = kwargs['log']
                with open(trainer.config.save_folder+'log.txt', 'wt', encoding='utf-8') as file:
                    for key in log.keys():
                        file.writelines('{} : {}\n'.format(key, log[key]))