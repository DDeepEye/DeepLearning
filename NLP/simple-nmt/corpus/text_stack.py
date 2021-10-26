import sys

if __name__ == "__main__":
    target_1 = sys.argv[1]
    target_2 = sys.argv[2]
    output_file = sys.argv[3]    
    
    with open(sys.argv[1], 'rt', encoding='utf-8') as file:
        lines_1 = file.readlines()

    with open(sys.argv[2], 'rt', encoding='utf-8') as file:
        lines_2 = file.readlines()

    with open(output_file, 'wt', encoding='utf-8') as file:
        file.writelines(lines_1)
        file.writelines(lines_2)

