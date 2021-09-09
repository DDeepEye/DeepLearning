
import sys
import os
import argparse
import codecs

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="text file split split")

    parser.add_argument(
        '--input', '-i', type=str,required=True,
        metavar='PATH',
        help="Input text (default: standard input). output file path => input+_a,  input+'_b'")
    parser.add_argument(
        '--ratio', '-r', type=float, default=0.9,
        help="original size = ratio, remainder size = 1 - ratio")

    return parser


if __name__ == "__main__":
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

    parser = create_parser()
    args = parser.parse_args()

    lines_a = []
    lines_b = []
    with open(args.input, 'rt', encoding='UTF8') as file:
        lines = file.readlines()
        length = len(lines)
        boundary = length*args.ratio
        lines_a = lines[:boundary]
        lines_b = lines[boundary:]

    with open(args.input+'_a', 'wt', encoding='UTF8') as file:
        file.writelines(lines_a)

    with open(args.input+'_b', 'wt', encoding='UTF8') as file:
        file.writelines(lines_b)
    


    


