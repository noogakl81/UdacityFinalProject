import input_files
import pandas as pd
import re
import collections
import math
import sys

def make_idf_dictionaries(df_list, constant_idf_factor):
    word_idf = make_inverse_document_frequency_dictionary(df_list, constant_idf_factor)
    bigram_idf = make_inverse_document_bigram_frequency_dictionary(df_list, constant_idf_factor)
    return word_idf, bigram_idf

# function to make idf for words
def make_inverse_document_frequency_dictionary(df_list, constant_idf_factor):
    idf_dict = {}
    total_words = set()
    for df in df_list:
        nQueries = len(df)
        for i in range(nQueries):
# don't want to double count words, so we use a set to represent them for idf
            words = set(df['content'].iloc[i].split())
            words = words.union(set(df['title'].iloc[i].split()))
# summing up weighted count of words
            for word in words:
                if word not in idf_dict:
                    idf_dict[word] = (constant_idf_factor/nQueries)
                else:
                    idf_dict[word] += (constant_idf_factor/nQueries)
                total_words.add(word)
# taking logarithm of sum of weighted count of words
    for word in idf_dict:
        nOccurences = idf_dict[word]
        idf_dict[word] = math.log(max(constant_idf_factor*len(df_list)/nOccurences-1,1e-3)) 
    return idf_dict

# function to make idf for bigrams
def make_inverse_document_bigram_frequency_dictionary(df_list, constant_idf_factor):
    bigram_idf_dict = {}
    total_bigrams = set()
    for df in df_list:
        nQueries = len(df)
        for i in range(nQueries):
            words = df['content'].iloc[i].split()
            nwords = len(words)
# don't want to double count bigrams, so we use a set to represent bigrams currently in use
            current_bigrams = set()
# summing up weighted count of bigrams in content 
          for j in range(nwords-1):
                bigram = words[j] + ' ' + words[j+1]
                if bigram not in bigram_idf_dict:
                    bigram_idf_dict[bigram] = (constant_idf_factor/nQueries)
                    current_bigrams.add(bigram)
                elif bigram not in current_bigrams:
                    bigram_idf_dict[bigram] += (constant_idf_factor/nQueries)
                    current_bigrams.add(bigram)
                total_bigrams.add(bigram)
            words = df['title'].iloc[i].split()
            nwords = len(words)
# summing up weighted count of bigrams in title
            for j in range(nwords-1):
                bigram = words[j] + ' ' + words[j+1]
                if bigram not in bigram_idf_dict:
                    bigram_idf_dict[bigram] = (constant_idf_factor/nQueries)
                    current_bigrams.add(bigram)
                elif bigram not in current_bigrams:
                    bigram_idf_dict[bigram] += (constant_idf_factor/nQueries)
                    current_bigrams.add(bigram)
                total_bigrams.add(bigram)
# taking logarithm of sum of weighted count of bigrams
    for bigram in bigram_idf_dict:
        nOccurences = bigram_idf_dict[bigram]
        bigram_idf_dict[bigram] = math.log(max(constant_idf_factor*len(df_list)/nOccurences-1,1e-3)) 
    return bigram_idf_dict

# function to make tf dictionary for word
def make_term_frequency_dictionary(query_df, word_df, constant_idf_factor):
    nQueries = len(query_df)
    word_dict = {}
    for i in range(nQueries):
# counting words in content
        words = query_df['content'].iloc[i].split()
        for word in words:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
# counting words in title
        words = query_df['title'].iloc[i].split()
        for word in words:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
# taking log of count
    for word in word_dict:
        word_dict[word] = 1.0 + max(math.log(word_dict[word]),0)
    word_df['tf'] = pd.Series(word_dict)

# function to make tf dictionary for bigrams
def make_bigram_frequency_dictionary(query_df,bigram_df,constant_idf_factor):
    nQueries = len(query_df)
    bigram_dict = {}
    for i in range(nQueries):
        words = query_df['content'].iloc[i].split()
        nwords = len(words)
# counting bigrams in content
        for j in range(nwords-1):
            bigram = words[j] + ' ' + words[j+1]
            if bigram not in bigram_dict:
                bigram_dict[bigram] = 1
            else:
                bigram_dict[bigram] += 1
        words = query_df['title'].iloc[i].split()
        nwords = len(words)
# counting bigrams in title
        for j in range(nwords-1):
            bigram = words[j] + ' ' + words[j+1]
            if bigram not in bigram_dict:
                bigram_dict[bigram] = 1
            else:
                bigram_dict[bigram] += 1
# taking log of counts
    for bigram in bigram_dict:
        bigram_dict[bigram] = 1.0 + max(math.log(bigram_dict[bigram]),0)
    bigram_df['tf'] = pd.Series(bigram_dict)

# function creates tf-idf columns for words and bigrams, and adds columns to bigram dataframe for tf-idf for each word in bigram
def get_bigram_word_dfs(input_df,word_df,bigram_df,constant_idf_factor):
    make_term_frequency_dictionary(input_df, word_df, constant_idf_factor)
    output_word_df = pd.DataFrame(word_df.dropna())
    output_word_df.name = input_df.name + '_word'
    output_word_df['tfidf'] = output_word_df['tf']*output_word_df['idf']

    make_bigram_frequency_dictionary(input_df, bigram_df, constant_idf_factor)

    output_bigram_df = pd.DataFrame(bigram_df.dropna())
    output_bigram_df.name = input_df.name + '_bigram'
    output_bigram_df['tfidf'] = output_bigram_df['tf']*output_bigram_df['idf']
    output_bigram_df['bigram_index'] = output_bigram_df.index
    output_bigram_df['word_1'] = output_bigram_df['bigram_index'].apply(lambda x: x.split()[0])
    output_bigram_df['word_2'] = output_bigram_df['bigram_index'].apply(lambda x: x.split()[1])

    output_bigram_df = pd.merge(output_bigram_df,pd.DataFrame({'tfidf1' : output_word_df['tfidf']}),left_on='word_1',right_index=True,sort=False)

    output_bigram_df = pd.merge(output_bigram_df,pd.DataFrame({'tfidf2' : output_word_df['tfidf']}),left_on='word_2',right_index=True,sort=False)

    output_bigram_df.drop('bigram_index',axis=1,inplace=True)
    output_bigram_df.drop('word_1',axis=1,inplace=True)
    output_bigram_df.drop('word_2',axis=1,inplace=True)

    return output_bigram_df,output_word_df

if __name__ == "__main__":
    import time
    t1 = time.time()

    biology_df  = pd.read_csv('biology_clean.csv')
    cooking_df  = pd.read_csv('cooking_clean.csv')
    crypto_df   = pd.read_csv('crypto_clean.csv')
    diy_df      = pd.read_csv('diy_clean.csv')
    robotics_df = pd.read_csv('robotics_clean.csv')
    travel_df   = pd.read_csv('travel_clean.csv')
    test_df     = pd.read_csv('test_clean.csv')

    if len(sys.argv) == 2:
        f = float(sys.argv[1])
        print f
        if f < 1.0:
            biology_df  = pd.DataFrame(biology_df.sample(frac=f,random_state=29))
            cooking_df  = pd.DataFrame(cooking_df.sample(frac=f,random_state=29))
            crypto_df   = pd.DataFrame(crypto_df.sample(frac=f,random_state=29))
            diy_df      = pd.DataFrame(diy_df.sample(frac=f,random_state=29))
            robotics_df = pd.DataFrame(robotics_df.sample(frac=f,random_state=29))
            travel_df   = pd.DataFrame(travel_df.sample(frac=f,random_state=29))
            test_df     = pd.DataFrame(test_df.sample(frac=f,random_state=29))

    biology_df.name  = 'biology'
    cooking_df.name  = 'cooking'
    crypto_df.name   = 'crypto'
    diy_df.name      = 'diy'
    robotics_df.name = 'robotics'
    travel_df.name   = 'travel'
    test_df.name     = 'test'

    t2 = time.time()
    print " time to read in data files = %s " % (t2-t1)

    constant_idf_factor = 1000.0
    train_df = [biology_df, cooking_df, crypto_df, diy_df, robotics_df, travel_df]
#    train_df = [biology_df]
    all_df = train_df + [test_df]
    
    for df in train_df:
        df = input_files.remove_dashes_from_tags(df)

    word_document_frequency = make_inverse_document_frequency_dictionary(all_df, constant_idf_factor)
    bigram_document_frequency = make_inverse_document_bigram_frequency_dictionary(all_df, constant_idf_factor)

    t3 = time.time()
    print " time to remove dashes and make idf = %s " % (t3-t2)

    word_df   = pd.DataFrame({'idf' : pd.Series(word_document_frequency)})
    bigram_df = pd.DataFrame({'idf' : pd.Series(bigram_document_frequency)})

    word_df.index.name = 'word'
    bigram_df.index.name = 'bigram'

    biology_bigram_df,biology_word_df = get_bigram_word_dfs(biology_df,word_df,bigram_df,constant_idf_factor)
    cooking_bigram_df,cooking_word_df = get_bigram_word_dfs(cooking_df,word_df,bigram_df,constant_idf_factor)
    crypto_bigram_df,crypto_word_df = get_bigram_word_dfs(crypto_df,word_df,bigram_df,constant_idf_factor)
    diy_bigram_df,diy_word_df = get_bigram_word_dfs(diy_df,word_df,bigram_df,constant_idf_factor)
    robotics_bigram_df,robotics_word_df = get_bigram_word_dfs(robotics_df,word_df,bigram_df,constant_idf_factor)
    travel_bigram_df,travel_word_df = get_bigram_word_dfs(travel_df,word_df,bigram_df,constant_idf_factor)
    test_bigram_df,test_word_df = get_bigram_word_dfs(test_df,word_df,bigram_df,constant_idf_factor)

    t4 = time.time()
    print " time to make word and bigram tf-idf dataframes = %s " % (t4-t3)

    biology_bigram_df.to_csv((biology_df.name + '_bigram.csv'),header=True,index=True)
    biology_word_df.to_csv((biology_df.name + '_word.csv'),header=True,index=True)

    cooking_bigram_df.to_csv((cooking_df.name + '_bigram.csv'),header=True,index=True)
    cooking_word_df.to_csv((cooking_df.name + '_word.csv'),header=True,index=True)

    crypto_bigram_df.to_csv((crypto_df.name + '_bigram.csv'),header=True,index=True)
    crypto_word_df.to_csv((crypto_df.name + '_word.csv'),header=True,index=True)

    diy_bigram_df.to_csv((diy_df.name + '_bigram.csv'),header=True,index=True)
    diy_word_df.to_csv((diy_df.name + '_word.csv'),header=True,index=True)

    robotics_bigram_df.to_csv((robotics_df.name + '_bigram.csv'),header=True,index=True)
    robotics_word_df.to_csv((robotics_df.name + '_word.csv'),header=True,index=True)

    travel_bigram_df.to_csv((travel_df.name + '_bigram.csv'),header=True,index=True)
    travel_word_df.to_csv((travel_df.name + '_word.csv'),header=True,index=True)

    test_bigram_df.to_csv((test_df.name + '_bigram.csv'),header=True,index=True)
    test_word_df.to_csv((test_df.name + '_word.csv'),header=True,index=True)

    t5 = time.time()
    print " time to write word and bigram tf-idf dataframes = %s " % (t5-t4)
