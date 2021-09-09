import sys
import os

import torch
import torch.nn as nn
import torchtext
version = list(map(int, torchtext.__version__.split('.')))
if version[0] <= 0 and version[1] < 9:
    from torchtext import data
else:
    from torchtext.legacy import data

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from Arguments import NLPArgument
from ModelHeaders.rnn import RNNClassifier
from ModelHeaders.cnn import CNNClassifier

def read_text(max_length=256):
    '''
    Read text from standard input for inference.
    '''
    lines = []

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split(' ')[:max_length]]

    return lines

def define_field():
    '''
    To avoid use DataLoader class, just declare dummy fields. 
    With those fields, we can retore mapping table between words and indice.
    '''
    return (
        data.Field(
            use_vocab=True,
            batch_first=True,
            include_lengths=False,
        ),
        data.Field(
            sequential=False,
            use_vocab=True,
            unk_token=None,
        )
    )

if __name__ == '__main__':
    saved_data = torch.load(
        sys.argv[1],
        map_location='cpu' if int(sys.argv[2]) < 0 else 'cuda:%d' % int(sys.argv[2])
    )

    arg:NLPArgument = saved_data['config']

    rnn_model = saved_data['rnn'] if arg.use_rnn else None
    cnn_model = saved_data['cnn'] if arg.use_cnn else None

    vocab = saved_data['vocab']
    classes = saved_data['classes']

    vocab_size = len(vocab)
    n_classes = len(classes)

    text_field, label_field = define_field()

    print('vacab size => {}'.format(len(vocab)))

    text_field.vocab = vocab
    label_field.vocab = classes

    lines = read_text(max_length=arg.max_length)

    with torch.no_grad():
        ensemble = []
        if rnn_model is not None:
            # Declare model and load pre-trained weights.
            model = RNNClassifier(
                input_size=vocab_size,
                word_vec_size=arg.word_vec_size,
                hidden_size=arg.hidden_size,
                n_classes=n_classes,
                n_layers=arg.layer_number,
                dropout_p=arg.dropout,
            )
            model.load_state_dict(rnn_model)
            ensemble += [model]
        if cnn_model is not None:
            # Declare model and load pre-trained weights.
            model = CNNClassifier(
                input_size=vocab_size,
                word_vec_size=arg.word_vec_size,
                n_classes=n_classes,
                use_batch_norm=arg.use_batch_norm,
                dropout_p=arg.dropout,
                window_sizes=arg.window_sizes,
                n_filters=arg.filter_sizes,
            )
            model.load_state_dict(cnn_model)
            ensemble += [model]

        y_hats = []
        # Get prediction with iteration on ensemble.
        for model in ensemble:
            if arg.gpu_id >= 0:
                model.cuda(arg.gpu_id)
            # Don't forget turn-on evaluation mode.
            model.eval()

            y_hat = []
            for idx in range(0, len(lines), arg.batch_size):                
                # Converts string to list of index.
                x = text_field.pad(lines[idx:idx + arg.batch_size])
                print('pad => {}'.format(x))
                x = text_field.numericalize(
                    x,
                    device='cuda:%d' % arg.gpu_id if arg.gpu_id >= 0 else 'cpu',
                )
                print('numericalize => {}'.format(x))

                y_hat += [model(x).cpu()]
            # Concatenate the mini-batch wise result
            y_hat = torch.cat(y_hat, dim=0)
            # |y_hat| = (len(lines), n_classes)

            y_hats += [y_hat]

            model.cpu()
        # Merge to one tensor for ensemble result and make probability from log-prob.
        y_hats = torch.stack(y_hats).exp()
        # |y_hats| = (len(ensemble), len(lines), n_classes)
        y_hats = y_hats.sum(dim=0) / len(ensemble) # Get average
        # |y_hats| = (len(lines), n_classes)

        probs, indice = y_hats.topk(1)

        for i in range(len(lines)):
            sys.stdout.write('%s\t%s\n' % (
                ' '.join([classes.itos[indice[i][j]] for j in range(1)]), 
                ' '.join(lines[i])
            ))
