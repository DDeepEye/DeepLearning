import sys
import torch

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def read_text(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()

        labels, texts = [], []
        for line in lines:
            if line.strip() != '':
                # The file should have tab delimited two columns.
                # First column indicates label field,
                # and second column indicates text field.
                label, text = line.strip().split('\t')
                labels += [label]
                texts += [text]

    return labels, texts


def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.grad.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm

def eos_insert(x:list, eos:int)->list:
    x = [x[0], x[1]]
    batch_size = x[0].size(0)
    max_length = x[0].size(1)

    for i in range(batch_size):
        length = x[1][i]
        if max_length > length:
            x[0][i,length] = eos
            x[1][i] += 1
    return x


def bos_insert(x:list, bos:int)->list:
    x = [x[0], x[1]]
    batch_size = x[0].size(0)
    bos_x = x[0].new_zeros(batch_size, 1) + bos
    x[0] = torch.cat([bos_x, x[0][:,:-1]], dim=1)
    x[1] += 1
    return x


def bos_remove(x:list, pad:int)->list:
    x = [x[0], x[1]]
    batch_size = x[0].size(0)
    bos_x = x[0].new_zeros(batch_size, 1) + pad
    x[0] = torch.cat([x[0][:,1:], bos_x], dim=1)
    x[1] -= 1
    return x
    
