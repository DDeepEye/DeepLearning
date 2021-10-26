import sys

TWO_STR = '▁▁'
TRANS_STR = '§'

if __name__ == "__main__":
    target_file = sys.argv[1]
    output_file = sys.argv[2]
    with open(sys.argv[1], 'rt', encoding='utf-8') as file:
        lines = file.readlines()

    for index, line in enumerate(lines):
        lines[index] = line.replace(TWO_STR, TRANS_STR)

    with open(sys.argv[2], 'wt', encoding='utf-8') as file:
        file.writelines(lines)
