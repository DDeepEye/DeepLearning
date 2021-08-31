
import sys

STR = '▁'
TWO_STR = '▁▁'

def detokenization(line):
    if TWO_STR in line:
        line = line.strip().replace(' ', '').replace(TWO_STR, ' ').replace(STR, '').strip()
    else:
        line = line.strip().replace(' ', '').replace(STR, ' ').strip()

    return line


if __name__ == "__main__":
    lines = []
    with open(sys.argv[1],'rt', encoding='UTF8') as file:
        lines = file.readlines()
        file.close()

    length = len(lines)
    print('read line number = {}'.format(length))

    newLines = []    
    with open(sys.argv[2],'w', encoding='UTF8') as file:
        cnt = 0
        for line in lines:
            cnt += 1
            newLines.append(detokenization(line)+'\n')
            if cnt % 10000 == 0 or cnt == length : print("{0:.0%}".format(cnt / length))

        file.writelines(newLines)