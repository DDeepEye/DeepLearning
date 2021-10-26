import sys

if __name__ == "__main__":
    target_file = sys.argv[1]
    output_file = sys.argv[2]
    beginline = int(sys.argv[3])
    offset = int(sys.argv[4])
    with open(sys.argv[1], 'rt', encoding='utf-8') as file:
        lines = file.readlines()

    lines = lines[beginline:beginline+offset]

    with open(sys.argv[2], 'wt', encoding='utf-8') as file:
        file.writelines(lines)


