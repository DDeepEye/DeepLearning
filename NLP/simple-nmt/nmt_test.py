
import sys
import torch

import argparse
import pprint

from torch.serialization import load
from torchtext.legacy import vocab

from simple_nmt.data_loader import DataLoader
from simple_nmt.models.seq2seq import Seq2Seq

STR = '▁'
TWO_STR = '▁▁'

def detokenization(line:str)->str:
    if TWO_STR in line:
        line = line.strip().replace(' ', '').replace(TWO_STR, ' ').replace(STR, '').strip()
    else:
        line = line.strip().replace(' ', '').replace(STR, ' ').strip()

    return line



if __name__ == '__main__':
    saved_data = torch.load(
        sys.argv[1]
        ,map_location='cpu'
    )

    config = saved_data['config']
    saved_model = saved_data['model']
    src_vocab = saved_data['src_vocab']
    tgt_vocab = saved_data['tgt_vocab']

    loader = DataLoader()
    loader.load_vocab(src_vocab, tgt_vocab)

    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
    model = Seq2Seq(
            input_size,
            config.word_vec_size,           # Word embedding vector size
            config.hidden_size,             # LSTM's hidden vector size
            output_size,
            n_layers=config.n_layers,       # number of layers in LSTM
            dropout_p=config.dropout        # dropout-rate in LSTM
        )

    model.load_state_dict(saved_model)    

    print('src_vocab size ={}'.format(len(loader.src.vocab)))
    print('tgt_vocab size ={}'.format(len(loader.tgt.vocab)))
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)
    print(model)


    with open(sys.argv[2], 'rt', encoding='utf-8') as file:
        lines = file.readlines()
    
    line = lines[0].strip()[:config.max_length]
    print(line)
    paded = loader.src.pad(line)
    print(len(paded[0]))
    nums = loader.src.numericalize(paded)
    nums = nums[0].contiguous().view(1,-1)
    print(nums.shape)
    print(nums)







    

    



    

    