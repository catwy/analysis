import os
import re
import csv
import json
import time
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


def clean_null(df, cols, null_words):
    # To remove rows if a certain column contains elements in the list of null words
    # df is a loaded csv file, i.e. df = pd.read_csv(file.csv)
    print 'Before clean_null:', len(df['Name'])
    for x in null_words:
        df = df[df[cols] != x]
    print 'After clean_null:', len(df['Name'])
    return df

def split_two(df, dic):
    for key, value in dic.iteritems():
        df['temp1'],df[value[0]] = df[key].str.split("u'text':").str
        df[value[0]], df['temp1'] = df[value[0]].str.split("', u'link':").str
        df[value[0]] = df[value[0]].str.replace("u'",'')
        df['temp1'],df[value[1]] = df[key].str.split("u'link':").str
        df[value[1]] = df[value[1]].str.replace("u'",'')
        df[value[1]], df['temp1'] = df[value[1]].str.split("'\}").str
        df = df.drop('temp1', 1)
    return df


def clean_up(s):
    ascii_part = [c for c in s if ord(c) < 128]
    x = ''.join(ascii_part).strip()
    return ' '.join(x.split())

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

def date_yr_mon(df, dic):
    for key, value in dic.iteritems():
        df[key] = df[key].str.replace('(?:-).*','')
        df[key] = df[key].str.replace("00,","01,")
        df.ix[df[key].str.startswith('(\d+)'), key] = "January" + df[key].astype(str)
        df[key+'_test'] = pd.to_datetime(df[key], errors='coerce')
        df[value[0]] = df[key + '_test'].apply(lambda x: x.year)
        df[value[1]] = df[key + '_test'].apply(lambda x: x.month)
        df = df.drop([key,key+'_test'], 1)
    return df

def state_county_city(df):
    df['count'] = df['Parent'].str.count('>')
    df['c1'], df['c2'], df['c3'], df['c4'], df['c5'], df['c6'], df['c7'] = df['Parent'].str.split('>').str
    dic = {6: ['c3', 'c5', 'c6'], 5: ['c3', 'c5', 'c5'], 4: ['c3', 'c4', 'c4']}
    df['State'] = df['County'] = df['City'] = ""
    for key, value in dic.iteritems():
        df.ix[df['count'] == key, 'State'] = df[value[0]]
        df.ix[df['count'] == key, 'County'] = df[value[1]]
        df.ix[df['count'] == key, 'City'] = df[value[2]]
    list = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7','Parent','count']
    df = df.drop(list, 1)
    return df

def split_votes_share(df, dic):
    for key, value in dic.iteritems():
        df[value[0]], df[value[1]] = df[key].str.split("(").str
        df[value[0]] = df[value[0]].apply(clean_up)
        df[value[1]], df['temp'] = df[value[1]].str.split("%").str
        df = df.drop('temp', 1)
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
    start = time.time()
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

    list = dics.keys()+['Filing Deadline','Last Modified', 'Polls Open',
                        'v1','v2','v3','v4','v5','v6','v7']
    df = df.drop(list,1)

    dic = {'Offices':'Office', 'Polls Closes':'Polls Close', "Term Starts":"Term Start",
           'Term Ends':'Term End','Types':'Type'}
    for key, value in dic.iteritems():
        df = df.rename(columns={key: value})

    list = ['ContributorID']
    for x in list:
        df[x] = df[x].str.extract('(\d+)', expand = False)

    df['Source'] = df['Source'].str.replace('\[Link\]',"")

    df.loc[df['Type']=="", 'Type'] = df['Turnout']
    df['Turnout'] = df['Turnout'].str.extract('(\d+.\d+)', expand = False).astype(float)

    dic = {"Term Start": ["Term Start Year", "Term Start Month"],
           "Term End": ["Term End Year", "Term End Month"],
           "Polls Close": ["Poll Year", "Poll Month"]}
    df = date_yr_mon(df,dic)

    df = state_county_city(df)

    df.to_csv("test3.csv")

    print df.head(10)
    end = time.time()
    print("Race Details 1 is finished", end-start, 'elapsed')


    #======================================================#
    #    Data Frame Setup and Cleaning for Race Details 2  #
    #======================================================#
    start = time.time()
    df = pd.read_csv('key_race_details2.csv')
    df = clean_null(df, 'Name', ["{u'text': u'', u'link': u''}"])

    dics = {'Name':['Names','CandID'],
            'Certified Votes':['Votes_Share','v1'],
            'Party':['Partys', 'PartyID'],
            'Website':['v2','Web']}
    df = split_two(df, dics)

    list = ['CandID','PartyID']
    for x in list:
        df[x] = df[x].str.extract('(\d+)', expand = False)

    dics = {'Votes_Share':['Votes','Share']}
    df = split_votes_share(df,dics)

    list = ["Votes_Share", "Photo","Entry Date", "Margin", "Predict Avg.",
            "Cash On Hand", "Name", "Certified Votes", "Party", "Website", "v1", "v2"]
    df = df.drop(list,1)

    dic = {'Names':'Name', 'Partys':'Party'}
    for key, value in dic.iteritems():
        df = df.rename(columns={key: value})

    df.to_csv("test4.csv")

    print df.head(13)
    end = time.time()
    print ("Race Details 2 is finished", end-start, 'elapsed')












