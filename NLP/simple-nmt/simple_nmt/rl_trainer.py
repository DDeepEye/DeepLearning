import sys
import os

from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

import numpy as np

import torch
from torch.nn import functional as F
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast

import simple_nmt.data_loader as data_loader
from simple_nmt.trainer import BaseTrainer

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))))
from Utilitis.utilitis import get_grad_norm, get_parameter_norm, printProgress

class MinimumRiskTrainer(BaseTrainer):

    @staticmethod
    def _get_reward(y_hat, y, n_gram=6, method='gleu'):
        # This method gets the reward based on the sampling result and reference sentence.
        # For now, we uses GLEU in NLTK, but you can used your own well-defined reward function.
        # In addition, GLEU is variation of BLEU, and it is more fit to reinforcement learning.
        sf = SmoothingFunction()
        score_func = {
            'gleu':  lambda ref, hyp: sentence_gleu([ref], hyp, max_len=n_gram),
            'bleu1': lambda ref, hyp: sentence_bleu([ref], hyp,
                                                    weights=[1./n_gram] * n_gram,
                                                    smoothing_function=sf.method1),
            'bleu2': lambda ref, hyp: sentence_bleu([ref], hyp,
                                                    weights=[1./n_gram] * n_gram,
                                                    smoothing_function=sf.method2),
            'bleu4': lambda ref, hyp: sentence_bleu([ref], hyp,
                                                    weights=[1./n_gram] * n_gram,
                                                    smoothing_function=sf.method4),
        }[method]

        # Since we don't calculate reward score exactly as same as multi-bleu.perl,
        # (especialy we do have different tokenization,) I recommend to set n_gram to 6.

        # |y| = (batch_size, length1)
        # |y_hat| = (batch_size, length2)

        with torch.no_grad():
            scores = []

            for b in range(y.size(0)):
                ref, hyp = [], []
                for t in range(y.size(-1)):
                    ref += [str(int(y[b, t]))]
                    if y[b, t] == data_loader.EOS:
                        break

                for t in range(y_hat.size(-1)):
                    hyp += [str(int(y_hat[b, t]))]
                    if y_hat[b, t] == data_loader.EOS:
                        break
                # Below lines are slower than naive for loops in above.
                # ref = y[b].masked_select(y[b] != data_loader.PAD).tolist()
                # hyp = y_hat[b].masked_select(y_hat[b] != data_loader.PAD).tolist()

                scores += [score_func(ref, hyp) * 100.]
            scores = torch.FloatTensor(scores).to(y.device)
            # |scores| = (batch_size)

            return scores

    @staticmethod
    def _get_loss(y_hat, indice, reward=1):
        # |indice| = (batch_size, length)
        # |y_hat| = (batch_size, length, output_size)
        # |reward| = (batch_size,)
        batch_size = indice.size(0)
        output_size = y_hat.size(-1)

        '''
        # Memory inefficient but more readable version
        mask = indice == data_loader.PAD
        # |mask| = (batch_size, length)
        indice = F.one_hot(indice, num_classes=output_size).float()
        # |indice| = (batch_size, length, output_size)
        log_prob = (y_hat * indice).sum(dim=-1)
        # |log_prob| = (batch_size, length)
        log_prob.masked_fill_(mask, 0)
        log_prob = log_prob.sum(dim=-1)
        # |log_prob| = (batch_size, )
        '''

        # Memory efficient version
        log_prob = -F.nll_loss(
            y_hat.view(-1, output_size),
            indice.view(-1),
            ignore_index=data_loader.PAD,
            reduction='none'
        ).view(batch_size, -1).sum(dim=-1)

        loss = (log_prob * -reward).sum()
        # Following two equations are eventually same.
        # \theta = \theta - risk * \nabla_\theta \log{P}
        # \theta = \theta - -reward * \nabla_\theta \log{P}
        # where risk = -reward.

        return loss

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

            # Take sampling process because set False for is_greedy.
            y_hat, indice = self.model.search(
                x,
                is_greedy=False,
                max_length=self.config.max_length
            )

            with torch.no_grad():
                # Based on the result of sampling, get reward.
                actor_reward = MinimumRiskTrainer._get_reward(
                    indice,
                    y,
                    n_gram=self.config.rl_n_gram,
                    method=self.config.rl_reward,
                )
                # |y_hat| = (batch_size, length, output_size)
                # |indice| = (batch_size, length)
                # |actor_reward| = (batch_size)

                # Take samples as many as n_samples, and get average rewards for them.
                # I figured out that n_samples = 1 would be enough.
                baseline = []

                for _ in range(self.config.rl_n_samples):
                    _, sampled_indice = self.model.search(
                        x,
                        is_greedy=False,
                        max_length=self.config.max_length,
                    )
                    baseline += [
                        MinimumRiskTrainer._get_reward(
                            sampled_indice,
                            y,
                            n_gram=self.config.rl_n_gram,
                            method=self.config.rl_reward,
                        )
                    ]

                baseline = torch.stack(baseline).mean(dim=0)
                # |baseline| = (n_samples, batch_size) --> (batch_size)

                # Now, we have relatively expected cumulative reward.
                # Which score can be drawn from actor_reward subtracted by baseline.
                reward = actor_reward - baseline
                # |reward| = (batch_size)

            # calculate gradients with back-propagation
            loss = MinimumRiskTrainer._get_loss(
                y_hat,
                indice,
                reward=reward
            )
            backward_target = loss.div(y.size(0)).div(self.config.iteration_per_update)
            backward_target.backward()

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
                self.optimizer.step()
            

            fix = 'loss : {:.2f}  actor reward : {:.2f}  baseline : {:.2f}  reward : {:.2f} |param| : {:.4f}  |g_param| : {:.4f}'\
                .format(float(loss), float(actor_reward.mean()), float(baseline.mean()), float(reward.mean()), p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0., g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.)

            printProgress(index, len(self.train_loader), prefix=fix)

        return {
            'actor': float(actor_reward.mean()),
            'baseline': float(baseline.mean()),
            'reward': float(reward.mean()),
            '|param|': p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
            '|g_param|': g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.,
        }

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

                # feed-forward
                _, indice = self.model.search(
                    x,
                    is_greedy=True,
                    max_length=self.config.max_length,
                )
                # |y_hat| = (batch_size, length, output_size)
                # |indice| = (batch_size, length)
                reward = MinimumRiskTrainer._get_reward(
                    indice,
                    y,
                    n_gram=self.config.rl_n_gram,
                    method=self.config.rl_reward,
                )
            fix = 'reward : {:.2f}'.format(float(reward.mean()))
            printProgress(index, len(self.valid_loader), prefix=fix)
        return {
            'BLEU': float(reward.mean()),
        }
    
    def _next_epoch(self):
        self.config.rl_init_epoch += 1
        return self.config.rl_init_epoch

    def _get_epochs(self):
        max_epochs = self.config.rl_n_epochs - self.config.rl_init_epoch
        max_iteration = max_epochs * len(self.train_loader)
        return self.config.rl_init_epoch, self.config.rl_n_epochs, max_epochs ,max_iteration

    def _warnning(self):
        print('warnning!!! 강화 학습의 현재 epoch {} 까지 모든 학습이 끝났습니다. \n학습을 연장하고 싶으시면 epoch 설정을 다시 하세요'.format(self.config.n_epochs))