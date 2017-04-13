import csv
import evaluate_model
import input_files
import pandas as pd
import re
import sys
import tf_idf

def replace_spaces_with_dashes(tags):
    new_tags = set()
    for tag in tags:
        if len(tag.split()) == 1:
            new_tags.add(tag)
        else:
            new_tags.add(tag.replace(' ','-'))
    return new_tags

def predict_physics_tags(x, *params):
    word_threshold_value,bigram_threshold_value,word_bigram_threshold_value = x
    test_df,test_word_df,test_bigram_df,constant_idf_factor = params

    word_threshold = test_word_df['tfidf'].mean() + word_threshold_value*test_word_df['tfidf'].std()
    bigram_threshold = test_bigram_df['tfidf'].mean() + bigram_threshold_value*test_bigram_df['tfidf'].std()
    word_bigram_threshold = test_word_df['tfidf'].mean() + word_bigram_threshold_value*test_word_df['tfidf'].std()

    predicted_tags2 = set(test_bigram_df[(test_bigram_df['tfidf'] > bigram_threshold) & (test_bigram_df['tfidf1'] > word_bigram_threshold) & (test_bigram_df['tfidf2'] > word_bigram_threshold)].index.values)
    predicted_tags  = set(test_word_df[test_word_df['tfidf'] > word_threshold].index.values)

    test_df.index.name = 'id'

    test_df['predicted_tags'] = test_df.apply(lambda row:evaluate_model.get_tags_from_query(predicted_tags,predicted_tags2,row['content'],row['title']),axis=1)
    test_df['tags'] = test_df.apply(lambda row:replace_spaces_with_dashes(row['predicted_tags']),axis=1)

    fileName = 'predicted_physics_tags.csv'
    nRecords = len(test_df)

# writing out test predictions in format specified by Kaggle
    f = open(fileName, 'w')
    f.write("id,tags\n")
    for i in range(0,nRecords):
        filestring = str(test_df['id'].iloc[i]) + ','
        for tag in test_df['tags'].iloc[i]:
            if len(tag) > 1:
                filestring += tag.replace(' ','-') + ' '
            else:
                filestring += tag + ' '
        filestring += '\n'
        f.write(filestring)
    f.close()

if __name__ == "__main__":
    import time
    t1 = time.time()

    if len(sys.argv) == 4:
        word_threshold_value = float(sys.argv[1])
        bigram_threshold_value = float(sys.argv[2])
        word_bigram_threshold_value = float(sys.argv[3])
    else:
        print " usage: evaluate_model word_threshold bigram_threshold word_bigram_threshold "
        sys.exit()

# reading in cleaned test data, test bigram dataframe and test word dataframe
    test_df      = pd.read_csv('test_clean.csv',index_col=0)
    test_df.name = 'test'
    bigram_type = {'bigram':str,'idf':float,'tf':float,'tfidf':float,'tfidf1':float,'tfidf2':float}
    word_type = {'word':str,'idf':float,'tf':float,'tfidf':float}
    test_bigram_df = pd.read_csv('test_bigram.csv',dtype=bigram_type,index_col=0)
    test_word_df = pd.read_csv('test_word.csv',dtype=word_type,index_col=0)
    test_bigram_df.name = 'test_bigram'
    test_word_df.name = 'test_word'

    constant_idf_factor = 1000.0

# predicting physics tags
    x = (word_threshold_value,bigram_threshold_value,word_bigram_threshold_value)
    params = (test_df,test_word_df,test_bigram_df,constant_idf_factor)

    predict_physics_tags(x, *params)

    t2 = time.time()

    print "time to make physics predictions = %s " % (t2-t1)
