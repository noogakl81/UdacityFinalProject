
import input_files
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
import numpy as np
import pandas as pd
import pylab
import time

# returns fraction of tags which are actually found in content / title of stack exchange query
def get_percent_of_tags_in_title_question(df):
    count = 0;
    for tag in df["tags"]:
        if (" "+tag+" ") in (" " + df["title"] + " "):
            count += 1
        elif (" "+tag+" ") in (" " + df["content"] + " "):
            count += 1
    return float(count)/df['n_tags']

# returns number of tags which only contain one word
def get_word_count_tags1(df):
    count1 = 0
    for tag in df["tags"]:
        nwords_in_tag = len(tag.split());
        if nwords_in_tag == 1:
            count1 += 1
    return count1

# returns number of tags which only contain two words
def get_word_count_tags2(df):
    count2 = 0
    for tag in df["tags"]:
        nwords_in_tag = len(tag.split());
        if nwords_in_tag == 2:
            count2 += 1
    return count2

# returns number of tags which only contain three words
def get_word_count_tags3(df):
    count3 = 0
    for tag in df["tags"]:
        nwords_in_tag = len(tag.split());
        if nwords_in_tag == 3:
            count3 += 1
    return count3

# returns number of tags which only contain four words
def get_word_count_tags4(df):
    count4 = 0
    for tag in df["tags"]:
        nwords_in_tag = len(tag.split());
        if nwords_in_tag == 4:
            count4 += 1
    return count4

# makes histogram of distribution of number of tags for training data
def make_tag_count_histogram(biology_df,cooking_df,crypto_df,diy_df,robotics_df,travel_df):
    pylab.rcParams['figure.figsize'] = (14, 8)
    df =pd.DataFrame({'Biology Tag Count Distribution':biology_df['n_tags'],'Cooking Tag Count Distribution':cooking_df['n_tags'],'Crypto Tag Count Distribution':crypto_df['n_tags'],'DIY Tag Count Distribution':diy_df['n_tags'],'Robotics Tag Count Distribution':robotics_df['n_tags'],'Travel Tag Count Distribution':travel_df['n_tags']})
    plt2 = df.hist(layout=(2,3),range=[0.5,5.5],grid=False,bins=5,edgecolor='white')
    for i in range(0,2):
        for j in range(0,3):
            plt2[i][j].set_xlabel("Number of tags")
            plt2[i][j].set_ylabel("Number of queries")
    plt.savefig('TagCountDistribution.pdf', bbox_inches='tight',dpi='figure')

# makes histogram of percentage of tags actually found in content / title
def make_percentage_found_distribution_plot(biology_df,cooking_df,crypto_df,diy_df,robotics_df,travel_df):
    pylab.rcParams['figure.figsize'] = (14, 8)
    df =pd.DataFrame({'Biology':biology_df['percent_tag_found'],'Cooking':cooking_df['percent_tag_found'],'Crypto':crypto_df['percent_tag_found'],'DIY':diy_df['percent_tag_found'],'Robotics':robotics_df['percent_tag_found'],'Travel':travel_df['percent_tag_found']})
    plt1 = df.hist(layout=(2,3),range=[0,1],grid=False,bins=5,edgecolor='white')
    for i in range(0,2):
        for j in range(0,3):
            plt1[i][j].set_xlabel("Percentage of tags found in title/query")
            plt1[i][j].set_ylabel("Number of queries")
    plt.savefig('PercentageFoundDistribution.pdf', bbox_inches='tight',dpi='figure')

# makes histogram of distribution of tags with given number of words
def make_tag_word_number_distribution_plot(biology_df,cooking_df,crypto_df,diy_df,robotics_df,travel_df):
    total_nQueries = 0;
    j = 0
    one_percentage = []
    two_percentage = []
    three_percentage = []
    four_percentage = []
    indices_percentage = []
    for df in [biology_df, cooking_df, crypto_df, diy_df, robotics_df, travel_df]:
        nQueries = len(df)
        df['n_tags'] = df['tags'].apply(lambda x: len(x))
        df['n_title_words'] = df['title'].apply(lambda x: len(x.split()))
        df['n_query_words'] = df['content'].apply(lambda x: len(x.split()))
        df['percent_tag_found'] = df.apply(get_percent_of_tags_in_title_question, axis=1)
        df['one_word_tags'] = df.apply(get_word_count_tags1,axis=1)
        df['two_word_tags'] = df.apply(get_word_count_tags2,axis=1)
        df['three_word_tags'] = df.apply(get_word_count_tags3,axis=1)
        df['four_word_tags'] = df.apply(get_word_count_tags4,axis=1)
        print "Mean number of words: %f " % (pd.Series.mean(df['n_query_words']))
        print "Max number of words: %f " % (pd.Series.max(df['n_query_words']))
        print "Min number of words: %f " % (pd.Series.min(df['n_query_words']))    
        print "Mean number of tags: %f " % (pd.Series.mean(df['n_tags']))
        print "Mean percentage of tags found: %f " % (pd.Series.mean(df['percent_tag_found']))
        print "Max number of tags: %d " % (pd.Series.max(df['n_tags']))
        print "Min number of tags: %d " % (pd.Series.min(df['n_tags']))
        one_percentage   += [float(pd.Series.sum(df['one_word_tags']))/pd.Series.sum(df['n_tags'])]
        two_percentage   += [float(pd.Series.sum(df['two_word_tags']))/pd.Series.sum(df['n_tags'])]
        three_percentage += [float(pd.Series.sum(df['three_word_tags']))/pd.Series.sum(df['n_tags'])]
        four_percentage  += [float(pd.Series.sum(df['four_word_tags']))/pd.Series.sum(df['n_tags'])]
        indices_percentage += [df.name]

    pylab.rcParams['figure.figsize'] = (14, 8)
    d = {'1':one_percentage,'2':two_percentage,'3':three_percentage,'4':four_percentage}
    tag_word_count_df = pd.DataFrame(d, index=indices_percentage)
    print tag_word_count_df
    tag_word_count_plot = tag_word_count_df.plot(kind='bar',legend=True, fontsize=15)
    tag_word_count_plot.set_ylabel("Percentage of tags with given number of words",fontsize=15)
    tag_word_count_plot.legend(fontsize=15)
    plt.savefig('TagWordNumberDistribution.pdf', bbox_inches='tight',dpi='figure')

# makes histogram of word count distribution for training and test data
def make_word_count_query_distribution_plot(biology_df,cooking_df,crypto_df,diy_df,robotics_df,travel_df):
    pylab.rcParams['figure.figsize'] = (14, 8)
    df =pd.DataFrame({'Biology':biology_df['n_query_words'],'Cooking':cooking_df['n_query_words'],'Crypto':crypto_df['n_query_words'],'DIY':diy_df['n_query_words'],'Robotics':robotics_df['n_query_words'],'Travel':travel_df['n_query_words']})
    plt3 = df.hist(layout=(2,3),range=[0,500],grid=False,bins=50,edgecolor='white')
    for i in range(0,2):
        for j in range(0,3):
            plt3[i][j].set_xlabel("Number of words found in query")
            plt3[i][j].set_ylabel("Number of queries")
    plt.savefig('WordCountQueryDistribution.pdf', bbox_inches='tight',dpi='figure')

    pylab.rcParams['figure.figsize'] = (14, 6)
    df = physics_df['n_query_words']
    plt4 = df.hist(range=[0,500],grid=False,bins=50,edgecolor='white')
    plt4.set_xlabel("Number of words found in physics query")
    plt4.set_ylabel("Number of physics queries")
    plt.savefig('WordCountQueryTestDistribution.pdf', bbox_inches='tight',dpi='figure')

# make slice plot of harmonically averaged mean f1 scores of training data
def make_optimization_slice_plot():
    data = np.genfromtxt('extracted_data/slice.csv', skip_header=0, delimiter=',')
    y = data[1::4,0]
    x = data[0:4,1]

    f1_score = data[:,2].reshape(6,4)
    fig, ax = plt.subplots()
    cs = ax.contourf(x, y, f1_score)

    cbar = fig.colorbar(cs)
    cbar.set_label("Harmonically averaged mean f1 score")

    ax.set_xlabel("bigram threshold")
    ax.set_ylabel("word threshold")
    ax.set_title("word bigram threshold = 0.5")
    plt.savefig('OptimizationSlice.pdf',bbox_inches='tight')

if __name__ == "__main__":
    t1 = time.time()

    biology_df  = pd.read_csv('biology_clean.csv')
    cooking_df  = pd.read_csv('cooking_clean.csv')
    crypto_df   = pd.read_csv('crypto_clean.csv')
    diy_df      = pd.read_csv('diy_clean.csv')
    robotics_df = pd.read_csv('robotics_clean.csv')
    travel_df   = pd.read_csv('travel_clean.csv')
    test_df     = pd.read_csv('test_clean.csv')

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
    all_df = train_df + [test_df]
    
    for df in train_df:
        df = input_files.remove_dashes_from_tags(df)

    make_tag_word_number_distribution_plot(biology_df,cooking_df,crypto_df,diy_df,robotics_df,travel_df)
    make_percentage_found_distribution_plot(biology_df,cooking_df,crypto_df,diy_df,robotics_df,travel_df)
    make_tag_count_histogram(biology_df,cooking_df,crypto_df,diy_df,robotics_df,travel_df)
    make_optimization_slice_plot()
