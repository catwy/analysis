import os
import re
import csv
import json
import collections
import numpy as np
import pandas as pd
from collections import defaultdict

# Merge Race Details
def json2csv(count, file1, csvs):
    # file1 is a loaded json file
    file1 = collections.OrderedDict(sorted(file1.items()))
    file2 = open(csvs, 'a')
    csvwriter = csv.writer(file2)
    if count == 0: # write columns
       header = file1.keys()
       csvwriter.writerow(header)
       count += 1
    if count >= 1: # skip columns
       csvwriter.writerow(file1.values())
    file2.close()

# Merge Candidates votes in a Race
def json2csv2(count, file1, csvs, key):
    # file1 is a loaded json file
    file1 = collections.OrderedDict(sorted(file1.items()))
    file2 = open(csvs, 'a')
    csvwriter = csv.writer(file2)
    if count == 0:
        header = file1.keys()
        csvwriter.writerow(header)
        count += 1
    if count >= 1:
        length1 = len(file1)  # number of keys
        length2 = len(file1[key]) # number of observations
        for l2 in range(length2):
            row2 = []
            for l1 in range(length1):
                row1 = file1.values()[l1]
                row2.append(row1[l2])
            csvwriter.writerow(row2)
    file2.close()

def key_race_details(folder, office, csvs):
    #if os.path.exists(csvs):
    #    os.remove(csvs)
    count = 0
    for file in os.listdir(folder):
        filename = os.path.basename(file)
        if ("race_" in filename) & (office in filename) & ("RaceDetails" in filename):
            x = re.findall(r'\d+', str(filename))
            i = x[1]
            with open(os.path.join(folder, file)) as f2:
                son = json.load(f2)
                son['RaceID'] = i
            json2csv(count, son, csvs)
            count += 1
    f2.close()

def key_race_details2(folder, office, csvs, key):
    #if os.path.exists(csvs):
    #    os.remove(csvs)
    count = 0
    for file in os.listdir(folder):
        filename = os.path.basename(file)
        if ("race_" in filename) & (office in filename) & ("Candidates" in filename):
            x = re.findall(r'\d+', str(filename))
            i = x[1]
            print('race id', i)
            with open(os.path.join(folder, file)) as f2:
                son = json.load(f2)
                w = len(son[key])
                son['RaceID'] = np.repeat(i, [w])
            json2csv2(count, son, csvs, key)
            count += 1
    f2.close()


def extract_1int(v):
    # find the number in string v, suited for one integer only
    x = re.findall(r'\d+', v)
    w = None
    if x:
       w = x[0]
    return w

def dropcols(df, list):
    for x in list:
        df = df.drop(x,1)
    return df

def clean_null(df, cols, null_words):
    # To remove rows if a certain column contains elements in the list of null words
    # df is a loaded csv file, i.e. df = pd.read_csv(file.csv)
    print 'Before clean_null:', len(df['Name'])
    for x in null_words:
        df = df[df[cols] != x]
    print 'After clean_null:', len(df['Name'])
    return df

def remove_u(v):
    # clean up the u'kk' into kk
    v1 = v.split("u'",1)[1]
    v2 = v1.split("'",1)[0]
    return v2

def split_left(v):
    # decompose the string v into two parts and keep the left
    v = str(v)
    left = ""
    if (v is not "") & (v != "nan"):
        left = v.split("u'text':", 1)[1]
        left = remove_u(left)
    return left

def split_right(v):
    # decompose the string v into two parts and keep the left
    v = str(v)
    right = ""
    if (v is not "") & (v != "nan"):
        right = v.split("u'link':", 1)[1]
        right = remove_u(right)
    return right

def split_two(df, dic):
    # apply split_left and split_right to selected columns in data frame df
    for key, value in dic.iteritems():
        df[value[0]] = df[key].apply(split_left)
        df[value[1]] = df[key].apply(split_right)
    return df

def clean_up(s):
    ascii_part = [c for c in s if ord(c) < 128]
    x = ''.join(ascii_part).strip()
    return ' '.join(x.split())

def split_votes(v):
    # split the votes and share apart and keep the "votes"
    v = str(v)
    votes = v.split("(",1)[0]
    votes = clean_up(votes)
    return votes

def split_share(v):
    v = str(v)
    if v != "":
        share = v.split("(",1)[1]
        share = share.split("%",1)[0]
    else:
        share = ""
    return share

def split_votes_share(df, dic):
    for key, value in dic.iteritems():
        df[value[0]] = df[key].apply(split_votes)
        df[value[1]] = df[key].apply(split_share)
    return df

def split_turnout(v):
    v = str(v)
    turnout = ""
    if "%" in v:
        turnout = v.split("%",1)[0]
    return turnout

def clean_csv(file):
    # This is to load a csv file with different number of columns in each row
    lines = list(csv.reader(open(file)))
    header, values = lines[0], lines[1:]
    w_header = len(header)
    w_lines = 0
    for x in lines[1:]:
        new = len(x)
        w_lines = max(w_lines, new)
    if w_lines > w_header:
        for x in range(w_lines - w_header):
            header.append("Append{}".format(x))
    df = pd.read_csv(file, names=header)
    df.drop(df.head(1).index, inplace=True)
    return df

if __name__ == '__main__':
    #======================================================#
    #    Initialize Directory and load Data                #
    #======================================================#

    dir0 = '/Users/yuwang/Documents/research/research/timing/git'
    dir1 = dir0 + '/analysis'
    dir2 = dir0 + '/campaigns/data'
    dir3 = dir0 + '/mayors/data'
    dir4 = dir0 + '/campaigns/schema'
    dir5 = dir0 + '/mayors/schema'

    '''
    # create a folder for cache
    if not os.path.exists('pdata'):
        os.mkdir('pdata')
    if os.path.exists('key_race_details.csv'):
        os.remove('key_race_details.csv')
    if os.path.exists('key_race_details2.csv'):
        os.remove('key_race_details2.csv')


    key_race_details(dir3, 'Mayor', 'key_race_details.csv')
    key_race_details2(dir3, 'Mayor', 'key_race_details2.csv', u'Certified Votes')
    '''

    #======================================================#
    #    Data Frame Setup and Cleaning for Race Details 1  #
    #======================================================#
    df = clean_csv('key_race_details.csv')

    dics = {'Contributor': ['Contributor Name', 'ContributorID'],
            'Data Sources': ['Source', 'Source Link'],
            'Office': ['Offices', 'v1'],
            'Parents': ['Parent', 'v2'],
            'Polls Close':['Polls Closes', 'v3'],
            'Term Start': ['Term Starts', 'v5'],
            'Term End':['Term Ends', 'v4'],
            'Type':['Turnout', 'v6'],
            'Append0':['Types','v7']}
    df = split_two(df, dics)

    list = dics.keys()+['Filing Deadline','Last Modified', 'Polls Open','v1','v2','v3','v4','v5','v6','v7']
    df = dropcols(df, list)

    dic = {'Offices':'Office', 'Polls Closes':'Polls Close', "Term Starts":"Term Start",
           'Term Ends':'Term End','Types':'Type'}
    for key, value in dic.iteritems():
        df = df.rename(columns={key: value})

    list = ['ContributorID']
    for x in list:
        df[x] = df[x].apply(extract_1int)

    df['Source'] = df['Source'].apply(lambda x: x.replace("[Link]",""))

    df.loc[df['Type']=="", 'Type'] = df['Turnout']
    df['Turnout'] = df['Turnout'].apply(lambda x: split_turnout(x))



    print df.head(10)


'''
    #======================================================#
    #    Data Frame Setup and Cleaning for Race Details 2  #
    #======================================================#

    df = pd.read_csv('key_race_details2.csv')
    df = clean_null(df, 'Name', ["{u'text': u'', u'link': u''}"])

    dics = {'Name':['Names','CandID'],
            'Certified Votes':['Votes_Share','v1'],
            'Party':['Partys', 'PartyID'],
            'Website':['v2','Web']}
    df = split_two(df, dics)

    list = ['CandID','PartyID']
    for x in list:
        df[x] = df[x].apply(extract_1int)

    dics = {'Votes_Share':['Votes','Share']}
    df = split_votes_share(df,dics)

    list = ["Votes_Share", "Photo","Entry Date", "Margin", "Predict Avg.",
            "Cash On Hand", "Name", "Certified Votes", "Party", "Website", "v1", "v2"]
    df = dropcols(df, list)

    dic = {'Names':'Name', 'Partys':'Party'}
    for key, value in dic.iteritems():
        df = df.rename(columns={key: value})

    print df.head()
'''











