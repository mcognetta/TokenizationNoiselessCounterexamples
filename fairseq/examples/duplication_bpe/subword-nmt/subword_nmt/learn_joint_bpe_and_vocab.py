#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
This script learns BPE jointly on a concatenation of a list of texts (typically the source and target side of a parallel corpus,
applies the learned operation to each and (optionally) returns the resulting vocabulary of each text.
The vocabulary can be used in apply_bpe.py to avoid producing symbols that are rare or OOV in a training text.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import unicode_literals

import sys
import os
import inspect
import codecs
import argparse
import tempfile
import warnings
import random
from collections import Counter
from multiprocessing import cpu_count

#hack to get imports working if running this as a script, or within a package
if __name__ == '__main__':
    import learn_bpe
    import apply_bpe
else:
    from . import learn_bpe
    from . import apply_bpe

# hack for python2/3 compatibility
from io import open
argparse.open = open

def create_parser(subparsers=None):

    if subparsers:
        parser = subparsers.add_parser('learn-joint-bpe-and-vocab',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")
    else:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), required=True, nargs = '+',
        metavar='PATH',
        help="Input texts (multiple allowed).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), required=True,
        metavar='PATH',
        help="Output file for BPE codes.")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s)")
    parser.add_argument(
        '--separator', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s')")
    parser.add_argument(
        '--write-vocabulary', type=argparse.FileType('w'), required=True, nargs = '+', default=None,
        metavar='PATH', dest='vocab',
        help='Write to these vocabulary files after applying BPE. One per input text. Used for filtering in apply_bpe.py')
    parser.add_argument(
        '--min-frequency', type=int, default=2, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s)')
    parser.add_argument(
        '--total-symbols', '-t', action="store_true",
        help="subtract number of characters from the symbols to be generated (so that '--symbols' becomes an estimate for the total number of symbols needed to encode text).")
    parser.add_argument(
        '--num-workers', type=int, default=20,
        help="Number of processors to process texts, only supported in Python3. If -1, set `multiprocessing.cpu_count()`. (default: %(default)s)")
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")

    # parser.add_argument('--random-drop-k', type=int, default = 0, help="the number of tokens to drop")
    # parser.add_argument('--random-drop-N', type=int, default = -1, help="the range to consider to drop from. -1 means consider the full vocab. must be > k")
    parser.add_argument('--duplication-n', type=int, default=-1, help='the top n to duplicate')
    parser.add_argument('--duplication-k', type=int, default=0, help='duplicate each token k times')
    return parser

def learn_joint_bpe_and_vocab(args):

    if args.vocab and len(args.input) != len(args.vocab):
        sys.stderr.write('Error: number of input files and vocabulary files must match\n')
        sys.exit(1)

    # read/write files as UTF-8
    args.input = [codecs.open(f.name, encoding='UTF-8') for f in args.input]
    args.vocab = [codecs.open(f.name, 'w', encoding='UTF-8') for f in args.vocab]

    # get combined vocabulary of all input texts
    full_vocab = Counter()
    full_character_vocab = set()
    for f in args.input:
        v, cv =  learn_bpe.get_vocabulary(f, num_workers=args.num_workers)
        full_vocab += v
        full_character_vocab = full_character_vocab | cv
        f.seek(0)

    # print(full_character_vocab)
    print(f"FULL CHARACTER VOCAB SIZE {len(full_character_vocab)}")
    # for c in full_vocab:
    #     full_vocab[c] += 1
    for c in full_character_vocab:
        # if c not in full_vocab:
        #     print(f"{c} not found in vocab")
        #     full_vocab[c] = 1

        # full_vocab[c] += args.character_default_increase
        pass

    vocab_list = ['{0} {1}'.format(key, freq) for (key, freq) in full_vocab.items()]

    # learn BPE on combined vocabulary
    with codecs.open(args.output.name, 'w', encoding='UTF-8') as output:
        learn_bpe.learn_bpe(vocab_list, output, args.symbols, args.min_frequency, args.verbose, is_dict=True, total_symbols=args.total_symbols)

    with codecs.open(args.output.name, encoding='UTF-8') as codes:
        bpe = apply_bpe.BPE(codes, separator=args.separator)

    appears_in_merge = set()
    with codecs.open(args.output.name, encoding='UTF-8') as codes:
        for line in codes:
            l, r = line.strip().split(' ')
            if '</w>' not in r:
                r = r + '@@'
            else:
                r = r[:-4]
            l = l + '@@'
            appears_in_merge.add(l)
            appears_in_merge.add(r)

    # apply BPE to each training corpus and get vocabulary
    for train_file, vocab_file in zip(args.input, args.vocab):

        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()

        tmpout = codecs.open(tmp.name, 'w', encoding='UTF-8')

        train_file.seek(0)
        bpe.process_lines(train_file.name, tmpout, num_workers=args.num_workers)

        tmpout.close()
        tmpin = codecs.open(tmp.name, encoding='UTF-8')

        vocab, character_vocab = learn_bpe.get_vocabulary(tmpin, num_workers=args.num_workers)
        tmpin.close()
        os.remove(tmp.name)

        for c in vocab:
            vocab[c] += 1
        for c in character_vocab:
            if c not in vocab:
                print(f"{c} not found in vocab")
                # vocab[c] = args.character_default_increase
                vocab[c] = 1
            else:
                # vocab[c] += args.character_default_increase
                vocab[c] += 1

        characters = [(x, y) for (x, y) in vocab.items() if x in character_vocab]
        subwords   = sorted([(x, y) for (x, y) in vocab.items() if x not in character_vocab], key = lambda x: x[1], reverse=True)
        full_vocab = characters + subwords

        for idx, (key, freq) in enumerate(sorted(full_vocab, key=lambda x: x[1], reverse=True)):

            vocab_file.write("{0} {1}\n".format(key, freq))
            if idx < args.duplication_n:
                for i in range(1, args.duplication_k + 1):
                    vocab_file.write(f"複複{i}複複{key} {freq}\n")

        train_file.close()
        vocab_file.close()


if __name__ == '__main__':

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    newdir = os.path.join(currentdir, 'subword_nmt')
    if os.path.isdir(newdir):
        warnings.warn(
            "this script's location has moved to {0}. This symbolic link will be removed in a future version. Please point to the new location, or install the package and use the command 'subword-nmt'".format(newdir),
            DeprecationWarning
        )

    # python 2/3 compatibility
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

    if args.num_workers <= 0:
        args.num_workers = cpu_count()

    if sys.version_info < (3, 0):
        args.separator = args.separator.decode('UTF-8')
        if args.num_workers > 1:
            args.num_workers = 1
            warnings.warn("Parallel mode is only supported in Python3. Using 1 processor instead.")

    assert(len(args.input) == len(args.vocab))

    learn_joint_bpe_and_vocab(args)
