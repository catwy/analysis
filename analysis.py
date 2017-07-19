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


def setup_race_details():
    start = time.time()
    df = clean_csv('key_race_details.csv')

    dics = {'Contributor': ['Contributor Name', 'ContributorID'],
            'Data Sources': ['Source', 'Source Link'],
            'Office': ['Offices', 'v1'],
            'Parents': ['Parent', 'v2'],
            'Polls Close': ['Polls Closes', 'v3'],
            'Term Start': ['Term Starts', 'v5'],
            'Term End': ['Term Ends', 'v4'],
            'Type': ['Turnout', 'v6'],
            'Append0': ['Types', 'v7']}
    df = split_two(df, dics)

    list = dics.keys() + ['Filing Deadline', 'Last Modified', 'Polls Open',
                          'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']
    df = df.drop(list, 1)

    dic = {'Offices': 'Office', 'Polls Closes': 'Polls Close', "Term Starts": "Term Start",
           'Term Ends': 'Term End', 'Types': 'Type'}
    for key, value in dic.iteritems():
        df = df.rename(columns={key: value})

    list = ['ContributorID']
    for x in list:
        df[x] = df[x].str.extract('(\d+)', expand=False)

    df['Source'] = df['Source'].str.replace('\[Link\]', "")

    df.loc[df['Type'] == "", 'Type'] = df['Turnout']
    df['Turnout'] = df['Turnout'].str.extract('(\d+.\d+)', expand=False).astype(float)

    dic = {"Term Start": ["Term Start Year", "Term Start Month"],
           "Term End": ["Term End Year", "Term End Month"],
           "Polls Close": ["Poll Year", "Poll Month"]}
    df = date_yr_mon(df, dic)

    df = state_county_city(df)

    df.to_csv("test3.csv")

    print df.head(10)
    end = time.time()
    print("Race Details 1 is finished", end - start, 'elapsed')
    return df


def setup_race_details2():
    start = time.time()
    df = pd.read_csv('key_race_details2.csv')
    df = clean_null(df, 'Name', ["{u'text': u'', u'link': u''}"])

    dics = {'Name': ['Names', 'CandID'],
            'Certified Votes': ['Votes_Share', 'v1'],
            'Party': ['Partys', 'PartyID'],
            'Website': ['v2', 'Web']}
    df = split_two(df, dics)

    list = ['CandID', 'PartyID']
    for x in list:
        df[x] = df[x].str.extract('(\d+)', expand=False)

    dics = {'Votes_Share': ['Votes', 'Share']}
    df = split_votes_share(df, dics)

    list = ["Votes_Share", "Photo", "Entry Date", "Margin", "Predict Avg.",
            "Cash On Hand", "Name", "Certified Votes", "Party", "Website", "v1", "v2"]
    df = df.drop(list, 1)

    dic = {'Names': 'Name', 'Partys': 'Party'}
    for key, value in dic.iteritems():
        df = df.rename(columns={key: value})

    df.to_csv("test4.csv")

    print df.head(13)
    end = time.time()
    print ("Race Details 2 is finished", end - start, 'elapsed')
    return df

def check_shares_sum():
    def add_shares():
        df_race2['Share'] = df_race2['Share'].astype(float)
        df = df_race2.groupby(['RaceID'])['Share'].sum().reset_index()
        df['Index'] = range(df.shape[0])
        df.to_csv("test5.csv")
        for x in [10, 50, 90, 98, 101, 1000]:
            print "<", x, len(df[(df['Share'] < x)])
        df = df[df['Share'] < 50]
        return df

    def shares_wrong_big():
        df['RaceID'] = df['RaceID'].astype(int)
        df_race['RaceID'] = df_race['RaceID'].astype(int)
        df2 = df_race.merge(df, left_on='RaceID', right_on='RaceID', how='outer')
        df3 = df_race2.merge(df2, left_on='RaceID', right_on='RaceID', how='outer')
        df3['Share_y'] = df3['Share_y'].astype(str)
        df3 = df3[df3['Share_y'].str.contains(r'\d+')]
        print df3.head(9)
        df3.to_csv("test6.csv")
        return df3

    def shares_wrong_small():
        g = df3['RaceID'].unique()
        s = pd.Series(g)
        s.to_csv('test7.csv')

    df = add_shares()
    df3 = shares_wrong_big()
    shares_wrong_small()
    return df3


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
    dir6 = dir0 + '/mayors'
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

    df_race = setup_race_details()
    df_race2 = setup_race_details2()
    df_shares_wrong = check_shares_sum()

    g = df_race2['CandID'].unique()
    s = pd.Series(g)
    print 'number of unique candidates=', len(s)
    s.to_csv('test8.csv')

    df = df_race2.groupby(['CandID'])['RaceID'].count().reset_index()
    df.to_csv('test9.csv')
    df = df[ (df['CandID']!='22593') & (df['CandID']!='191')] #write-in & others
    print df['RaceID'].describe()
    print df[df['RaceID']==16]
    df = df[ df['RaceID']>2]
    print df['RaceID'].describe()

    df_m1 = pd.read_csv('/Users/yuwang/Documents/research/research/timing/git/mayors/recent_elections_part1.txt', delimiter = ';', header = None)
    df_m2 = pd.read_csv('/Users/yuwang/Documents/research/research/timing/git/mayors/recent_elections_part2.txt', delimiter = ';', header = None)
    df_m3 = pd.read_csv('/Users/yuwang/Documents/research/research/timing/git/mayors/recent_elections_part3.txt', delimiter = ';', header = None)

    df_m = df_m1.append(df_m2)
    df_m = df_m.append(df_m3)
    h = df_m.shape[0]
    df_m['CityID'] = range(h)
    df_m.to_csv('messy.csv')
    dic = {0: 'web', 1: 'city', 2: "state",
           3: 'partisan', 4: 'note'}
    for key, value in dic.iteritems():
        df_m = df_m.rename(columns={key: value})

    print df_m.head()
    print 'number of cities with at least one election = ', len(df_m[df_m['web'].str.contains('http')])

    df_m['RaceID'] = df_m['web'].str.extract('(\d+)', expand = False)

    df_m = df_m.drop('note',1)
    df_m = df_m[df_m['web'].str.contains('http')]
    df_m['RaceID'] = df_m['RaceID'].astype(str)
    df_race['RaceID'] = df_race['RaceID'].astype(str)
    df = df_m.merge(df_race, left_on = 'RaceID', right_on = 'RaceID', how = 'outer')
    df.to_csv('test10.csv')

    df_city = df[['State', 'County', 'City', 'web', 'state', 'city', 'CityID']]
    df_city['CityID'] = df_city['CityID'].astype(str).copy()
    df_city = df_city[df_city['CityID'].str.contains('(\d+)')]
    df_city.to_csv('skeleton.csv')

    df_all = df_race.merge(df_city, left_on = ['State','City'], right_on = ['State', 'City'], how = 'outer')
    df_all.to_csv('all.csv')
    df_alls = df_all.groupby(['CityID'])['RaceID'].count().reset_index()
    print df_alls['RaceID'].describe()




















