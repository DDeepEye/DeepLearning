import argparse
import sys
import os
import codecs
from operator import itemgetter

import torch

from simple_nmt.data_loader import DataLoader
import simple_nmt.data_loader as data_loader
from simple_nmt.models.seq2seq import Seq2Seq

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from Arguments import NMTTranslateArg


STR = '▁'
TWO_STR = '▁▁'
TRANS_STR = '§'


def define_argparser(args : NMTTranslateArg):
    p = argparse.ArgumentParser()

    is_required = (args is None)

    p.add_argument(
        '--model_fn',
        required=is_required,
        help='Model file name to use'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to use. -1 for CPU. Default=%(default)s'
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Mini batch size for parallel inference. Default=%(default)s'
    )
    p.add_argument(
        '--max_length',
        type=int,
        default=255,
        help='Maximum sequence length for inference. Default=%(default)s'
    )
    p.add_argument(
        '--n_best',
        type=int,
        default=1,
        help='Number of best inference result per sample. Default=%(default)s'
    )
    p.add_argument(
        '--beam_size',
        type=int,
        default=5,
        help='Beam size for beam search. Default=%(default)s'
    )
    p.add_argument(
        '--lang',
        type=str,
        default=None,
        help='Source language and target language. Example: enko'
    )
    p.add_argument(
        '--length_penalty',
        type=float,
        default=1.2,
        help='Length penalty parameter that higher value produce shorter results. Default=%(default)s',
    )

    config = p.parse_args()

    if args is not None:
        config.model_fn = args.model_filepath
        config.batch_size = args.batch_size
        config.gpu_id = args.gpu_id

    return config


def read_text(batch_size=128):
    # This method gets sentences from standard input and tokenize those.
    lines = []

    sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split(' ')]

        if len(lines) >= batch_size:
            yield lines
            lines = []

    if len(lines) > 0:
        yield lines


def to_text(indice, vocab):
    # This method converts index to word to show the translation result.
    lines = []

    for i in range(len(indice)):
        line = []
        for j in range(len(indice[i])):
            index = indice[i][j]

            if index == data_loader.EOS:
                # line += ['<EOS>']
                break
            else:
                line += [vocab.itos[index]]

        line = ' '.join(line)
        lines += [line]

    return lines


def get_vocabs(saved_data):
    # Load vocabularies from the model.
    src_vocab = saved_data['src_vocab']
    tgt_vocab = saved_data['tgt_vocab']

    return src_vocab, tgt_vocab


def get_model(saved_data, input_size, output_size, train_config):
    model = Seq2Seq(
        input_size,
        train_config.word_vec_size,
        train_config.hidden_size,
        output_size,
        n_layers=train_config.n_layers,
        dropout_p=train_config.dropout,
    )

    model.load_state_dict(saved_data['model'])
    model.eval()
    return model


if __name__ == '__main__':    

    tr_args = NMTTranslateArg(
        model_filepath='./2021.0918.trans/NMT.pth'
        ,translate_filepath='./corpus/10_sampling.test.en'
        ,output_filepath='./corpus/translate.kr.txt'
        ,gpu_id=-1
        ,batch_size = 1
    )

    config = define_argparser(tr_args)

    # Load saved model.
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu',
    )

    # Load configuration setting in training.
    train_config = saved_data['config']

    src_vocab, tgt_vocab = get_vocabs(saved_data)

    # Initialize dataloader, but we don't need to read training & test corpus.
    # What we need is just load vocabularies from the previously trained model.
    loader = DataLoader()
    loader.load_vocab(src_vocab, tgt_vocab)

    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
    model = get_model(saved_data, input_size, output_size, train_config)

    # Put models to device if it is necessary.
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)

    with torch.no_grad():
        with open(tr_args.translate_filepath,'rt', encoding='utf-8') as tr:
            is_trans = True
            lines = []
            for i in range(tr_args.batch_size):
                line = tr.readline()
                if line == '':
                    break
                lines += [line.strip(' \n').split(' ')]

            if len(lines) == 0:
                is_trans = False

            while is_trans:
                # Get sentences from standard input.
                # Since packed_sequence must be sorted by decreasing order of length,
                # sorting by length in mini-batch should be restored by original order.
                # Therefore, we need to memorize the original index of the sentence.
                
                lengths         = [len(line) for line in lines]
                original_indice = [i for i in range(len(lines))]

                sorted_tuples = sorted(
                    zip(lines, lengths, original_indice),
                    key=itemgetter(1),
                    reverse=True,
                )
                sorted_lines    = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]
                lengths         = [sorted_tuples[i][1] for i in range(len(sorted_tuples))]
                original_indice = [sorted_tuples[i][2] for i in range(len(sorted_tuples))]

                paded = loader.src.pad(sorted_lines)
                print(paded)
                x = loader.src.numericalize(
                    paded,
                    device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu'
                )
                # |x| = (batch_size, length)

                y_hats, indice = model.search(x)
                # |y_hats| = (batch_size, length, output_size)
                # |indice| = (batch_size, length)

                output = to_text(indice, loader.tgt.vocab)
                sorted_tuples = sorted(zip(output, original_indice), key=itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                for i in range(len(output)):
                    output[i] = output[i].replace(' ','').replace(TWO_STR,' ').replace(STR, '').replace(TRANS_STR,' ')
                    output[i] += '\n'

                with open(tr_args.output_filepath, 'a', encoding='utf-8') as file:
                    file.writelines(output)

                lines = []
                for i in range(tr_args.batch_size):
                    line = tr.readline()
                    if line == '':
                        break                    
                    lines += [line.strip(' \n').split(' ')]

                if len(lines) == 0:
                    is_trans = False

