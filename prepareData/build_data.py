#!/usr/bin/env python3

from collections import OrderedDict
import fileinput
import sys
import pandas as pd

import numpy
import json

def get_word_vector(word_freqs,filename_list,index):


    for filename in filename_list:
        print('Processing', filename)
        with open(filename, 'r') as f:
            for line in f:
                # print('line is ',line )
                words_in = line.strip().split(',')[index]

                w = words_in
                if w not in word_freqs:
                    word_freqs[w] = 0
                word_freqs[w] += 1

    return word_freqs
def main():
        filename = []
        filename.append('./books/books_all_data.csv')
        # filename.append('.csv')
        df_s = pd.read_csv(filename[0])
        print(df_s.head())


        word_freqs = OrderedDict()
        #0 means user's ID, 1 means item's ID
        flag = 0
        word_freqs = get_word_vector(word_freqs,filename,flag)
        
        print(min(list(word_freqs.values())))
        words = list(word_freqs.keys())
        freqs = list(word_freqs.values())
        print(len(words))
        sorted_idx = numpy.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]

        worddict = OrderedDict()
        # FIXME We shouldn't assume <EOS>, <GO>, and <UNK> aren't BPE subwords.
        for ii, ww in enumerate(sorted_words):
            worddict[ww] = ii 

        # The JSON RFC requires that JSON text be represented using either
        # UTF-8, UTF-16, or UTF-32, with UTF-8 being recommended.
        # We use UTF-8 regardless of the user's locale settings.
        if flag == 0 :
            with open('books_userID.json', 'w') as f:
                json.dump(worddict, f, indent=2, ensure_ascii=False)
        else:
            with open('books_itemID.json', 'w') as f:
                json.dump(worddict, f, indent=2, ensure_ascii=False) 
        print('Done')

if __name__ == '__main__':
    main()
