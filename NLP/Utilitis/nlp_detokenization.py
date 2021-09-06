import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from Utilitis import utilitis

STR = '▁'
TWO_STR = '▁▁'

def detokenization(line):
    if TWO_STR in line:
        line = line.strip().replace(' ', '').replace(TWO_STR, ' ').replace(STR, '').strip()
    else:
        line = line.strip().replace(' ', '').replace(STR, ' ').strip()

    return line

if __name__ == "__main__":
    print("proc detok")
    with open(sys.argv[1],'rt', encoding='UTF8') as file:
        lines = file.readlines()
    
    length = len(lines)
    print('detokenzation size {}'.format(length))

    detokLines = []    
    print_frequence = int(length / 100)
    for i in range(length):
        detokLines.append(detokenization(lines[i])+'\n')
        if i % print_frequence == 0: utilitis.printProgress(i, length, 'detokenizabtion')

    with open(sys.argv[2], 'wt', encoding='UTF8') as file:
        file.writelines(detokLines)
    
    