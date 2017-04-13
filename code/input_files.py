import numpy as np
import pandas as pd
import re
import string
import unicodedata
from bs4 import BeautifulSoup

def get_clean_df_from_input_file(data_set_name, hasTags=True):
    df = read_input_csv_file('input/' + data_set_name + '.csv')
    df.name = data_set_name
    print df.name
    print "number of queries = %d " % (len(df))
    df = df_html_to_text(df)
    df = remove_punctuation_and_numbers_from_title_content(df)
    if hasTags:
        count_tags(df)
    count_text_words(df)
    return(df)

def read_input_csv_file(fileName):
    return pd.read_csv(fileName)

def convert_html_to_text(html_data):
    return BeautifulSoup(html_data,"lxml").get_text()

def df_html_to_text(df):
    df['content'] = df['content'].apply(convert_html_to_text)
    return df

# removing punctuation, hyperlinks, mathematical equations, and numbers

def remove_punctuation(s):
    s = ''.join([i if ord(i) < 128 else ' ' for i in s])
    s = str(s)
    s = re.sub("http.+\s|http.+$", " ", s)
    s = re.sub("www.+\s|www.+$", " ", s)
    t0 = re.compile(r"\$\$.*?\$\$",re.DOTALL)
    while re.search(t0,s):
        s = re.sub(t0, " ", s)
    t1 = re.compile(r"\$.*?[^\\{\[\]<>+=\-*_]+?.*?\$",re.DOTALL)
    while re.search(t1,s):
        s = re.sub(t1, " ", s)
    t2 = re.compile(r"\\begin{eq.*?}.*?\\end{eq.*?}",re.DOTALL)
    while re.search(t2,s):
        s = re.sub(t2, " ", s)
    t3 = re.compile(r"\\begin{gather}.*?\\end{gather}",re.DOTALL)
    while re.search(t3,s):
        s = re.sub(t3, " ", s)
    t4 = re.compile(r"\\begin{align.*?}.*?\\end{align.*?}",re.DOTALL)
    while re.search(t4,s):
        s = re.sub(t4, " ", s)
    blanktab = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    s = s.translate(blanktab)
    s = s.lower()
    t5 = re.compile(r'^[0-9]+\s|\s[0-9]+\s|\s[0-9]+$',re.DOTALL)
    while re.search(t5,s):
        s = re.sub(t5, " ", s)
    s = re.sub("\n", " ", s)
    s = re.sub("\s+", " ", s)
    return s

def break_up_string_into_words_and_clean(s):
    words = s.split()
    for idx,word in enumerate(words):
        words[idx] = word.replace("-", " ")
    return words

def remove_punctuation_and_numbers_from_title_content(df):
    for field in ['title','content']:
        df[field] = df[field].apply(remove_punctuation)
    return df

def remove_dashes_from_tags(df):
    df['tags'] = df['tags'].apply(break_up_string_into_words_and_clean)
    return df

def count_text_words(df):
    df['text_words'] = df['content'].apply(lambda x: len(str.split(x,' ')))
    print "Mean number of words: %f " % (pd.Series.mean(df['text_words']))
    print "Max number of words: %d " % (pd.Series.max(df['text_words']))
    print "Min number of words: %d " % (pd.Series.min(df['text_words']))
    df.drop('text_words',axis=1,inplace=True)

def count_tags(df):
    df['n_tags'] = df['tags'].apply(lambda x: len(str.split(x,' ')))
    print "Mean number of tags: %f " % (pd.Series.mean(df['n_tags']))
    print "Max number of tags: %d " % (pd.Series.max(df['n_tags']))
    print "Min number of tags: %d " % (pd.Series.min(df['n_tags']))
    df.drop('n_tags',axis=1,inplace=True)

if __name__ == "__main__":
    import time
    t1 = time.time()

    biology_df  = get_clean_df_from_input_file('biology')
    cooking_df  = get_clean_df_from_input_file('cooking')
    crypto_df   = get_clean_df_from_input_file('crypto')
    diy_df      = get_clean_df_from_input_file('diy')
    robotics_df = get_clean_df_from_input_file('robotics')
    travel_df   = get_clean_df_from_input_file('travel')
    test_df     = get_clean_df_from_input_file('test', hasTags = False)

    t2 = time.time()

    print "%s seconds to read data and clean it" % (t2-t1)

    biology_df.to_csv((biology_df.name + '_clean.csv'),header=True,index=True)
    cooking_df.to_csv((cooking_df.name + '_clean.csv'),header=True,index=True)
    crypto_df.to_csv((crypto_df.name + '_clean.csv'),header=True,index=True)
    diy_df.to_csv((diy_df.name + '_clean.csv'),header=True,index=True)
    robotics_df.to_csv((robotics_df.name + '_clean.csv'),header=True,index=True)
    travel_df.to_csv((travel_df.name + '_clean.csv'),header=True,index=True)
    test_df.to_csv((test_df.name + '_clean.csv'),header=True,index=True)

    t3 = time.time()

    print "%s seconds to read data and clean it" % (t3-t2)
