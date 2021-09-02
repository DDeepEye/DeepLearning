import sys
import os
import argparse
import codecs
import random
from tab_separate import tab_separate_data
from mecab_tokenization import tokenization_data
from cat import cat_list_to_list

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="tokenization")

    parser.add_argument(
        '--input', '-i', type=str,required=True,
        metavar='PATH',
        help="Input text (default: standard input).")
    parser.add_argument(
        '--output', '-o', type=str,required=True,
        metavar='PATH',
        help="Output file for BPE codes (default: standard output)")
    parser.add_argument(
        '--symbols', '-s', type=int, default=30000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s))")
    parser.add_argument(
        '--only_mecab', action='store_true',
        help='only mecab or bpe')

    parser.add_argument(
        '--is_shuffle', action='store_true',
        help='line to shuffle')

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

    with open(args.input, 'rt', encoding='UTF8') as file:
        lines = file.readlines()
        labels, texts = tab_separate_data(lines)

    texts = tokenization_data(texts, args.only_mecab)

    if args.only_mecab == False :
        temporary = 'temporary'
        modelfile = 'model'

        with open(temporary, 'wt', encoding='UTF8') as file:
            file.writelines(texts)

        proc = 'python .\subword-nmt/learn_bpe.py --input {} --output {} --symbols {}'.format(temporary, modelfile, args.symbols)
        print(proc)
        os.system(proc)

        proc = 'python .\subword-nmt/apply_bpe.py --codes {} --input {} --output {}'.format(modelfile, temporary, args.output)
        print(proc)
        os.system(proc)

        with open(args.output, 'rt', encoding='UTF8') as file:
            texts = file.readlines()

        os.remove(temporary)
        os.remove(modelfile)
    
    newlines = cat_list_to_list(labels , texts)

    random.shuffle(newlines)
    with open(args.output, 'wt', encoding='UTF8') as file:
        file.writelines(newlines)

