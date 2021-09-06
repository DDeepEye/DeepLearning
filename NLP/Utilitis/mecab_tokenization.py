
import sys
import os
import MeCab

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from Utilitis import utilitis

STR = '▁'

def tokenization_data(lines:list, only_mecab:bool) -> list:
    length = len(lines)
    print('read line number = {}'.format(length))

    m = MeCab.Tagger('-Owakati')
    newlines = []
    cnt = 0
    print_frequence = int(length / 100)
    for line in lines:
        
        line = line.strip()
        if only_mecab == False:
            line = line.replace(' ', ' '+STR)

        line = m.parse(line)

        if only_mecab == False:
            line = STR+line.replace(STR+' ', STR)

        newlines.append(line)

        cnt += 1
        if cnt % print_frequence == 0: utilitis.printProgress(cnt, length, 'mecab tokenizabtion')

    return newlines

def tokenization(src_path:str, tgt_path:str, only_mecab:bool):
    lines = []
    with open(src_path,'rt', encoding='UTF8') as file:
        lines = file.readlines()
        file.close()

    lines = tokenization_data(lines, only_mecab)

    with open(tgt_path,'a', encoding='UTF8') as file:
        file.writelines(lines)


if __name__ == "__main__":
    tokenization(sys.argv[1], sys.argv[2], sys.argv[3])
