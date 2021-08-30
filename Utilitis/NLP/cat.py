
import sys

def cat_line_to_line(src_a:str, src_b:str, ouputfile:str, segment = '\t'):
    with open(src_a,'rt', encoding='UTF8') as a:
        lines_a = a.readlines()
        length_a = len(lines_a)
    
    with open(src_b,'rt', encoding='UTF8') as b:
        lines_b = b.readlines()
        length_b = len(lines_b)

    newlines = []
    length = length_a
    if length_a != length_b:
        print('warning !!!! => Lines in two documents are not the same')
        length = length_a if length_a < length_b else length_b

    for i in range(length_a):
        newlines.append(lines_a[i].strip()+segment+lines_b[i])
        if i % 1000 == 0 or i == length : print("{0:.0%}".format(i / length))

    with open(ouputfile,'w', encoding='UTF8') as file:
        file.writelines(newlines)

if __name__ == "__main__":
    cat_line_to_line(sys.argv[1], sys.argv[2], sys.argv[3])



        