import math

import re
import sys
import time
import tf_idf
import input_files

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab

# computes precision for a given stack exchange query
def compute_precision_entry(predicted_tags,actual_tags):
    true_positives = 0
    for actual_tag in actual_tags:
        if actual_tag in predicted_tags:
            true_positives += 1
    false_positives = len(predicted_tags) - true_positives
    precision = float(true_positives)/(max(true_positives + false_positives,0.001))
    return precision

# computes recall for a given stack exchange query
def compute_recall_entry(predicted_tags,actual_tags):
    true_positives = 0
    for actual_tag in actual_tags:
        if actual_tag in predicted_tags:
            true_positives += 1
    false_negatives = len(actual_tags) - true_positives
    recall = float(true_positives)/(max(true_positives + false_negatives,0.001))
    return recall

#computes f1 score for a given stack exchange query
def compute_f1_score_entry(predicted_tags,actual_tags):
    true_positives = 0
    for actual_tag in actual_tags:
        if actual_tag in predicted_tags:
            true_positives += 1
    false_negatives = len(actual_tags) - true_positives
    false_positives = len(predicted_tags) - true_positives
    precision = float(true_positives)/(max(true_positives + false_positives,0.001))
    recall = float(true_positives)/(max(true_positives + false_negatives,0.001))
    f1_score = ((2 * precision * recall) / max(precision + recall,0.001))
    return f1_score

# returns full list of predicted tags for a given stack exchange query
def get_tags_from_query(predicted_tags, predicted_tags2, content, title):
    predicted_tags_entry = set()
    words = content.split()
    nwords = len(words)
    for j in range(nwords):
        if words[j] in predicted_tags:
            word = words[j]
            predicted_tags_entry.add(word)
        if j < (nwords-1):
            bigram = words[j] + ' ' + words[j+1]
            if bigram in predicted_tags2:
                predicted_tags_entry.add(bigram)
    words = title.split()
    nwords = len(words)
    for j in range(nwords):
        if words[j] in predicted_tags:
            word = words[j]
            predicted_tags_entry.add(word)
        if j < (nwords-1):
            bigram = words[j] + ' ' + words[j+1]
            if bigram in predicted_tags2:
                predicted_tags_entry.add(bigram)
    return predicted_tags_entry

# computes f1 scores for each training data set and returns the harmonic mean of the scores
def get_harmonic_mean_f1_scores(x, *params):
    train_df,train_word_df,train_bigram_df,constant_idf_factor = params
    word_threshold_value,bigram_threshold_value,word_bigram_threshold_value = x

    inverse_sum = 0

    f = open("results1.csv","a+")

    print x 
    fileString = str(word_threshold_value) + "," + str(bigram_threshold_value) + "," + str(word_bigram_threshold_value) + ","
    nTrainingSets = len(train_df)
    for i in range(0,nTrainingSets):
        word_threshold = train_word_df[i]['tfidf'].mean() + word_threshold_value*train_word_df[i]['tfidf'].std()
        bigram_threshold = train_bigram_df[i]['tfidf'].mean() + bigram_threshold_value*train_bigram_df[i]['tfidf'].std()
        word_bigram_threshold = train_word_df[i]['tfidf'].mean() + word_bigram_threshold_value*train_word_df[i]['tfidf'].std()
        predicted_tags2 = set(train_bigram_df[i].loc[(train_bigram_df[i]['tfidf'] > bigram_threshold) & (train_bigram_df[i]['tfidf1'] > word_bigram_threshold) & (train_bigram_df[i]['tfidf2'] > word_bigram_threshold)].index.values)
        predicted_tags = set(train_word_df[i].loc[train_word_df[i]['tfidf'] > word_threshold].index.values)

        train_df[i]['predicted_tags'] = train_df[i].apply(lambda row:get_tags_from_query(predicted_tags,predicted_tags2,row['content'],row['title']),axis=1)
        train_df[i]['precision'] = train_df[i].apply(lambda row:compute_precision_entry(row['predicted_tags'],row['tags']),axis=1)
        train_df[i]['recall'] = train_df[i].apply(lambda row:compute_recall_entry(row['predicted_tags'],row['tags']),axis=1)
        train_df[i]['f1_score'] = train_df[i].apply(lambda row:compute_f1_score_entry(row['predicted_tags'],row['tags']),axis=1)
        total_precision = train_df[i]['precision'].mean()
        total_recall = train_df[i]['recall'].mean()
        total_f1_score = train_df[i]['f1_score'].mean()

        if total_f1_score == 0:
            f.write(fileString + str(0) + '\n')
            f.close()
            return 0
        inverse_sum += 1./total_f1_score
        print train_df[i].name, total_precision, total_recall, total_f1_score
    harmonic_averaged_score = math.pow(inverse_sum/nTrainingSets,-1.0) 
    f.write(fileString + str(harmonic_averaged_score) +'\n')
    f.close()
    print "harmonic score %f" % (harmonic_averaged_score) 
    return -harmonic_averaged_score

if __name__ == "__main__":
    import time
    t1 = time.time()

    biology_df  = pd.read_csv('biology_clean.csv',index_col=0)
    cooking_df  = pd.read_csv('cooking_clean.csv',index_col=0)
    crypto_df   = pd.read_csv('crypto_clean.csv',index_col=0)
    diy_df      = pd.read_csv('diy_clean.csv',index_col=0)
    robotics_df = pd.read_csv('robotics_clean.csv',index_col=0)
    travel_df   = pd.read_csv('travel_clean.csv',index_col=0)
    test_df     = pd.read_csv('test_clean.csv',index_col=0)

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

    print " finished reading in data files time = %s " % (t2-t1)

    constant_idf_factor = 1000.0
    train_df = [biology_df, cooking_df, crypto_df, diy_df, robotics_df, travel_df]
    all_df = train_df + [test_df]
    
    for df in train_df:
        df = input_files.remove_dashes_from_tags(df)

    bigram_type = {'bigram':str,'idf':float,'tf':float,'tfidf':float,'tfidf1':float,'tfidf2':float}
    word_type = {'word':str,'idf':float,'tf':float,'tfidf':float}

# reading in word and bigram dataframes

    biology_bigram_df = pd.read_csv('biology_bigram.csv',dtype=bigram_type,index_col=0)
    biology_word_df = pd.read_csv('biology_word.csv',dtype=word_type,index_col=0)
    biology_bigram_df.name = 'biology_bigram'
    biology_word_df.name = 'biology_word'

    cooking_bigram_df = pd.read_csv('cooking_bigram.csv',dtype=bigram_type,index_col=0)
    cooking_word_df = pd.read_csv('cooking_word.csv',dtype=word_type,index_col=0)
    cooking_bigram_df.name = 'cooking_bigram'
    cooking_word_df.name = 'cooking_word'

    crypto_bigram_df = pd.read_csv('crypto_bigram.csv',dtype=bigram_type,index_col=0)
    crypto_word_df = pd.read_csv('crypto_word.csv',dtype=word_type,index_col=0)
    crypto_bigram_df.name = 'crypto_bigram'
    crypto_word_df.name = 'crypto_word'

    diy_bigram_df = pd.read_csv('diy_bigram.csv',dtype=bigram_type,index_col=0)
    diy_word_df = pd.read_csv('diy_word.csv',dtype=word_type,index_col=0)
    diy_bigram_df.name = 'diy_bigram'
    diy_word_df.name = 'diy_word'

    robotics_bigram_df = pd.read_csv('robotics_bigram.csv',dtype=bigram_type,index_col=0)
    robotics_word_df = pd.read_csv('robotics_word.csv',dtype=word_type,index_col=0)
    robotics_bigram_df.name = 'robotics_bigram'
    robotics_word_df.name = 'robotics_word'

    travel_bigram_df = pd.read_csv('travel_bigram.csv',dtype=bigram_type,index_col=0)
    travel_word_df = pd.read_csv('travel_word.csv',dtype=word_type,index_col=0)
    travel_bigram_df.name = 'travel_bigram'
    travel_word_df.name = 'travel_word'

    test_bigram_df = pd.read_csv('test_bigram.csv',dtype=bigram_type,index_col=0)
    test_word_df = pd.read_csv('test_word.csv',dtype=word_type,index_col=0)
    test_bigram_df.name = 'test_bigram'
    test_word_df.name = 'test_word'

    train_word_df = [biology_word_df, cooking_word_df, crypto_word_df, diy_word_df, robotics_word_df, travel_word_df]
    train_bigram_df = [biology_bigram_df, cooking_bigram_df, crypto_bigram_df, diy_bigram_df, robotics_bigram_df, travel_bigram_df]

# printing summary statistics on the training tf-idf data

    for word_df in train_word_df:
        print "%s tf-idf average = %lf" % (word_df.name, word_df['tfidf'].mean())
        print "%s tf-idf median = %lf" % (word_df.name, word_df['tfidf'].median())
        print "%s tf-idf stdev = %lf" % (word_df.name, word_df['tfidf'].std())

    for bigram_df in train_bigram_df:
        print "%s tf-idf average = %lf" % (bigram_df.name, bigram_df['tfidf'].mean())
        print "%s tf-idf median = %lf" % (bigram_df.name, bigram_df['tfidf'].median())
        print "%s tf-idf stdev = %lf" % (bigram_df.name, bigram_df['tfidf'].std())

# creating histogram plots for tf-idf distribution of training data

    for word_df in train_word_df:
        my_plot = word_df['tfidf'].plot.hist(bins=50)        
        filestring = word_df.name + '_histogram.pdf'
        pylab.savefig(filestring)
        plt.close()

    for bigram_df in train_bigram_df:
        my_plot = bigram_df['tfidf'].plot.hist(bins=50)
        filestring = bigram_df.name + '_histogram.pdf'
        pylab.savefig(filestring)
        plt.close()

# summary statistics for test data set

    print "%s tf-idf average = %lf" % (test_word_df.name, test_word_df['tfidf'].mean())
    print "%s tf-idf median = %lf" % (test_word_df.name, test_word_df['tfidf'].median())
    print "%s tf-idf stdev = %lf" % (test_word_df.name, test_word_df['tfidf'].std())

    print "%s tf-idf average = %lf" % (test_bigram_df.name, test_bigram_df['tfidf'].mean())
    print "%s tf-idf median = %lf" % (test_bigram_df.name, test_bigram_df['tfidf'].median())
    print "%s tf-idf stdev = %lf" % (test_bigram_df.name, test_bigram_df['tfidf'].std())

# histogram plot for tf-idf distribution of test data

    my_plot = test_word_df['tfidf'].hist(bins=50)        
    filestring = test_word_df.name + '_histogram.pdf'
    pylab.savefig(filestring)
    plt.clf()

    my_plot = test_bigram_df['tfidf'].hist(bins=50)
    filestring = test_bigram_df.name + '_histogram.pdf'
    pylab.savefig(filestring)
    plt.clf()

    t3 = time.time()

    print " finished reading bigram / word, doing stats and making plots time = %s " % (t3-t2)

    params = (train_df,train_word_df,train_bigram_df,constant_idf_factor)
    pranges = (slice(1.5,2.51,0.2),slice(3.5,5.1,0.5),slice(-0.5,2.1,0.5))

    from scipy import optimize

# calling optimizer

    minimum_result = optimize.brute(get_harmonic_mean_f1_scores, pranges, args=params, full_output=True, finish=optimize.fmin, disp=True)
# getting optimal values
    word_threshold_value = minimum_result[0][0]
    bigram_threshold_value = minimum_result[0][1]
    word_bigram_threshold_value = minimum_result[0][2]

    t4 = time.time()

    print " finished brute force optimization time = %s " % (t4-t3)

    print word_threshold_value, bigram_threshold_value, word_bigram_threshold_value

# writing out training data with predicted tags included in the file
    nTrainingSets = len(train_df)
    for i in range(0,nTrainingSets):
        word_threshold = train_word_df[i]['tfidf'].mean() + word_threshold_value*train_word_df[i]['tfidf'].std()
        bigram_threshold = train_bigram_df[i]['tfidf'].mean() + bigram_threshold_value*train_bigram_df[i]['tfidf'].std()
        word_bigram_threshold = train_word_df[i]['tfidf'].mean() + word_bigram_threshold_value*train_word_df[i]['tfidf'].std()
        predicted_tags2 = set(train_bigram_df[i].loc[(train_bigram_df[i]['tfidf'] > bigram_threshold) & (train_bigram_df[i]['tfidf1'] > word_bigram_threshold) & (train_bigram_df[i]['tfidf2'] > word_bigram_threshold)].index.values)
        predicted_tags  = set(train_word_df[i][train_word_df[i]['tfidf'] > word_threshold].index.values)

        train_df[i]['predicted_tags'] = train_df[i].apply(lambda row:get_tags_from_query(predicted_tags,predicted_tags2,row['content'],row['title']),axis=1)
        train_df[i]['precision'] = train_df[i].apply(lambda row:compute_precision_entry(row['predicted_tags'],row['tags']),axis=1)
        train_df[i]['recall'] = train_df[i].apply(lambda row:compute_recall_entry(row['predicted_tags'],row['tags']),axis=1)
        train_df[i]['f1_score'] = train_df[i].apply(lambda row:compute_f1_score_entry(row['predicted_tags'],row['tags']),axis=1)

        train_df[i].to_csv((train_df[i].name + '.csv'),header=True,index=True)
        train_word_df[i].to_csv((train_word_df[i].name + '.csv'),header=True,index=True)
        train_bigram_df[i].to_csv((train_bigram_df[i].name + '.csv'),header=True,index=True)

# writing out test data with predicted tags included in the file

    word_threshold = test_word_df['tfidf'].mean() + word_threshold_value*test_word_df['tfidf'].std()
    bigram_threshold = test_bigram_df['tfidf'].mean() + bigram_threshold_value*test_bigram_df['tfidf'].std()
    word_bigram_threshold = test_word_df['tfidf'].mean() + word_bigram_threshold_value*test_word_df['tfidf'].std()

    predicted_tags2 = set(test_bigram_df.loc[(test_bigram_df['tfidf'] > bigram_threshold) & (test_bigram_df['tfidf1'] > word_bigram_threshold) & (test_bigram_df['tfidf2'] > word_bigram_threshold)].index.values)
    predicted_tags  = set(test_word_df[test_word_df['tfidf'] > word_threshold].index.values)

    test_df['predicted_tags'] = test_df.apply(lambda row:get_tags_from_query(predicted_tags,predicted_tags2,row['content'],row['title']),axis=1)
    test_df.to_csv((test_df.name + '.csv'),header=True,index=True)
    test_word_df.to_csv((test_word_df.name + '.csv'),header=True,index=True)
    test_bigram_df.to_csv((test_bigram_df.name + '.csv'),header=True,index=True)

    print word_threshold, bigram_threshold, word_bigram_threshold
