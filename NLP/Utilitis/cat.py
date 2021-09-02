import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from Utilitis import utilitis

def cat_line_to_line(src_a:str, src_b:str, ouputfile:str, segment = '\t'):
    with open(src_a,'rt', encoding='UTF8') as a:
        lines_a = a.readlines()
    
    
    with open(src_b,'rt', encoding='UTF8') as b:
        lines_b = b.readlines()
    

    newlines = cat_list_to_list(lines_a, lines_b, segment)
    with open(ouputfile,'w', encoding='UTF8') as file:
        file.writelines(newlines)


def cat_list_to_list(lines_a:list, lines_b:list, segment='\t') -> list:
    length_a = len(lines_a)
    length_b = len(lines_b)

    length = length_a
    if length_a != length_b:
        print('warning !!!! => Lines in two documents are not the same')
        length = length_a if length_a < length_b else length_b

    newlines = []
    if length > 0:
        print_frequence = int(length / 100)
        for i in range(length):
            newlines.append(lines_a[i].strip(' \n')+segment+lines_b[i])
            if i % 100 == 0: utilitis.printProgress(i, length, 'list cat')

    return newlines


if __name__ == "__main__":
    cat_line_to_line(sys.argv[1], sys.argv[2], sys.argv[3])



        