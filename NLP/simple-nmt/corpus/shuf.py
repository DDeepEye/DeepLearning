import sys
import os
import random 

if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    if len(cur_dir) == 0 : cur_dir = '.'

    target_1 = cur_dir+'/corpus.shuf.train_valid_sum.tok.bpe.tr.en'
    target_2 = cur_dir+'/corpus.shuf.train_valid_sum.tok.bpe.tr.ko'
    output_file_a_1 = cur_dir+'/corpus.train.tr.en'
    output_file_a_2 = cur_dir+'/corpus.valid.tr.en'
    output_file_b_1 = cur_dir+'/corpus.train.tr.ko'
    output_file_b_2 = cur_dir+'/corpus.valid.tr.ko'
    lineratio = 0.9

    lines = []
    
    with open(target_1, 'rt', encoding='utf-8') as file:
        a = file.readlines()

    with open(target_2, 'rt', encoding='utf-8') as file:
        b = file.readlines()

    assert len(a) == len(b)

    for i ,data in enumerate(zip(a,b)):
        lines += [data]

    random.shuffle(lines)
    maxline = int(len(lines))
    train_maxline = int(maxline * lineratio)

    trainlines_a = []
    trainlines_b = []
    validlines_a = []
    validlines_b = []

    for i in range(train_maxline):
        trainlines_a += [lines[i][0]]
        trainlines_b += [lines[i][1]]

    for i in range(maxline - train_maxline):
        validlines_a += [lines[train_maxline+i][0]]
        validlines_b += [lines[train_maxline+i][1]]

    with open(output_file_a_1, 'wt', encoding='utf-8') as file:
        file.writelines(trainlines_a)
    with open(output_file_a_2, 'wt', encoding='utf-8') as file:
        file.writelines(validlines_a)
    with open(output_file_b_1, 'wt', encoding='utf-8') as file:
        file.writelines(trainlines_b)
    with open(output_file_b_2, 'wt', encoding='utf-8') as file:
        file.writelines(validlines_b)
        

