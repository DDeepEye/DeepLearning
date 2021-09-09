import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from Utilitis import utilitis

def tab_separate(src:str, output_lable:str, output_textdata:str):
    with open(src,'rt', encoding='UTF8') as file:
        lines = file.readlines()


    labels ,textdatas = tab_separate_data(lines)

    
    length = len(labels)
    for i in range(length):
        labels[i] = labels[i] + '\n'

    with open(output_lable, 'wt', encoding='UTF8') as file:
        file.writelines(labels)

    with open(output_textdata, 'wt', encoding='UTF8') as file:
        file.writelines(textdatas)

def tab_separate_data(lines:list):
    labels = []
    textdatas = []
    cnt = 0
    length = len(lines)
    print_frequence = int(length / 100)
    for line in lines:
        index = line.find('\t')
        labels.append(line[:index])
        textdatas.append(line[index+1:])
        cnt += 1
        if cnt % print_frequence == 0: utilitis.printProgress(cnt, length,'separate proc')

    length = len(labels)
    for i in range(length):
        labels[i] = labels[i] + '\n'

    return labels, textdatas
    
if __name__ == "__main__":
    tab_separate(sys.argv[1], sys.argv[2], sys.argv[3])