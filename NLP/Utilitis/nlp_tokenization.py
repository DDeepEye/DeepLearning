import sys
import os
import argparse
import codecs
import random
import time
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
        '--is_tabcat', action='store_true',
        help='line cat')
    parser.add_argument(
        '--is_shuffle', action='store_true',
        help='requere is_tabcat line to shuffle')

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

    labels = tokenization_data(labels, args.only_mecab) 
    texts = tokenization_data(texts, args.only_mecab)

    if args.only_mecab == False :
        corpusFiles = [labels, texts]

        cnt = 0
        for corpus in corpusFiles:
            temporary = args.output+'.temporary'
            modelfile = args.output+'.t{}'.format(cnt)+'.model'
            outputfile = args.output+'.t{}'.format(cnt)

            with open(temporary, 'wt', encoding='UTF8') as file:
                file.writelines(corpus)

            print('learn_bpe {} file'.format(temporary))
            proc = 'python .\subword-nmt/learn_bpe.py --input {} --output {} --symbols {} --verbose'.format(temporary, modelfile, args.symbols)
            print(proc)
            time.sleep(5)
            os.system(proc)

            print('apply_bpe {} file'.format(modelfile))
            proc = 'python .\subword-nmt/apply_bpe.py --codes {} --input {} --output {}'.format(modelfile, temporary, outputfile)
            print(proc)
            time.sleep(5)
            os.system(proc)

            with open(outputfile, 'rt', encoding='UTF8') as file:
                corpusFiles[cnt] = file.readlines()

            os.remove(temporary)
            cnt += 1
    
    if args.is_tabcat:
        newlines = cat_list_to_list(corpusFiles[0] , corpusFiles[1])
        if args.is_shuffle:
            random.shuffle(newlines)
        with open(args.output, 'wt', encoding='UTF8') as file:
            file.writelines(newlines)

