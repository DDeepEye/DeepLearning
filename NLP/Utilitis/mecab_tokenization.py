import MeCab
import sys
import os

STR = '▁'
def tokenization(src_path:str, tgt_path:str):
    lines = []
    with open(src_path,'rt', encoding='UTF8') as file:
        lines = file.readlines()
        file.close()

    length = len(lines)
    print('read line number = {}'.format(length))

    m = MeCab.Tagger('-Owakati')

    with open(tgt_path,'a', encoding='UTF8') as file:
        cnt = 0
        for line in lines:
            cnt += 1
            line = line.strip()
            line = line.replace(' ', ' '+STR)
            line = m.parse(line)
            line =STR+line.replace(STR+' ', STR)
            file.write(line)
            if cnt % 10000 == 0 or cnt == length : print("{0:.0%}".format(cnt / length))

if __name__ == "__main__":
    tokenization(sys.argv[1], sys.argv[2])
