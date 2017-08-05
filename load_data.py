import os
import re
import csv
import json
import time
import collections
import numpy as np
import pandas as pd


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

def key_race_details2(folder, office, csvs, key_new):
    count = 0
    for file in os.listdir(folder):
        filename = os.path.basename(file)
        if ("race_" in filename) & (office in filename) & ("Candidates" in filename):
            x = re.findall(r'\d+', str(filename))
            i = x[1]
            print('race id', i)
            with open(os.path.join(folder, file)) as f2:
                son = json.load(f2)
                for keys, values in son.items():
                    if "ertified Votes" in keys:
                        son[key_new] = son[keys]
                        del son[keys]
                w = len(son[key_new])
                son['RaceID'] = np.repeat(i, [w])
            json2csv2(count, son, csvs, key_new)
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
        df[key] = df[key].str.replace('(?:-).*','').str.replace("00,","01,")
        df.ix[df[key].str.startswith('(\d+)'), key] = "January" + df[key].astype(str)
        df[key+'_date'] = pd.to_datetime(df[key], errors='coerce')
        df[value[0]] = df[key + '_date'].apply(lambda x: x.year)
        df[value[1]] = df[key + '_date'].apply(lambda x: x.month)
        df = df.drop([key], 1)
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

def state(df):
    df['count'] = df['Parent'].str.count('>')
    print df['count'].max()
    df['c1'], df['c2'], df['State'], df['c4'], df['c5'], df['c6'] = df['Parent'].str.split('>').str
    list = ['c1', 'c2', 'c4', 'c5', 'c6','Parent','count']
    df = df.drop(list, 1)
    return df

def parent_split(df, dist):
    if dist == 'City':
        dm = state_county_city(df)
    else:
        dm = state(df)
    return dm


def split_votes_share(df, dic):
    for key, value in dic.iteritems():
        df[value[0]], df[value[1]] = df[key].str.split("(").str
        df[value[0]] = df[value[0]].apply(clean_up)
        df[value[1]], df['temp'] = df[value[1]].str.split("%").str
        df = df.drop('temp', 1)
    return df

def setup_race_details(dist):
    start = time.time()
    df = clean_csv('key_race_details.csv')

    if dist == 'City':
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

        df.loc[df['Type'] == "", 'Type'] = df['Turnout']
        df['Turnout'] = df['Turnout'].str.extract('(\d+.\d+)', expand=False).astype(float)
    else:
        dics = {'Contributor': ['Contributor Name', 'ContributorID'],
                'Data Sources': ['Source', 'Source Link'],
                'Office': ['Offices', 'v1'],
                'Parents': ['Parent', 'v2'],
                'Polls Close': ['Polls Closes', 'v3'],
                'Term Start': ['Term Starts', 'v5'],
                'Term End': ['Term Ends', 'v4'],
                'Turnout': ['Turnouts', 'v6'],
                'Type': ['Types', 'v7']}
        df = split_two(df, dics)

        list = dics.keys() + ['Filing Deadline', 'Last Modified', 'Polls Open',
                              'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']
        df = df.drop(list, 1)

        dic = {'Offices': 'Office', 'Polls Closes': 'Polls Close', "Term Starts": "Term Start",
               'Term Ends': 'Term End', 'Types': 'Type', 'Turnouts':'Turnout'}
        for key, value in dic.iteritems():
            df = df.rename(columns={key: value})

        #df.loc[df['Type'] == "", 'Type'] = df['Turnout']
        df['Turnout'] = df['Turnout'].str.extract('(\d+.\d+)', expand=False).astype(float)


    list = ['ContributorID']
    for x in list:
        df[x] = df[x].str.extract('(\d+)', expand=False)

    df['Source'] = df['Source'].str.replace('\[Link\]', "")

    dic = {"Term Start": ["Term Start Year", "Term Start Month"],
           "Term End": ["Term End Year", "Term End Month"],
           "Polls Close": ["Poll Year", "Poll Month"]}
    df = date_yr_mon(df, dic)

    df = parent_split(df,dist)

    df['Term Length'] = df['Term End Year'] - df['Term Start Year']

    print df.head(10)
    end = time.time()
    print("Race Details 1 is finished", end - start, 'elapsed')
    return df


def setup_race_details2():
    start = time.time()
    df = pd.read_csv('key_race_details2.csv')
    df = clean_null(df, 'Name', ["{u'text': u'', u'link': u''}"])

    dics = {'Name': ['Names', 'CandID'],
            'Final Votes': ['Votes_Share', 'v1'],
            'Party': ['Partys', 'PartyID'],
            'Website': ['v2', 'Web']}
    df = split_two(df, dics)

    list = ['CandID', 'PartyID']
    for x in list:
        df[x] = df[x].str.extract('(\d+)', expand=False)

    dics = {'Votes_Share': ['Votes', 'Share']}
    df = split_votes_share(df, dics)

    list = ["Votes_Share", "Photo", "Entry Date", "Margin", "Predict Avg.",
            "Cash On Hand", "Name", "Final Votes", "Party", "Website", "v1", "v2"]
    df = df.drop(list, 1)

    dic = {'Names': 'Name', 'Partys': 'Party'}
    for key, value in dic.iteritems():
        df = df.rename(columns={key: value})

    print df.head(13)
    end = time.time()
    print ("Race Details 2 is finished", end - start, 'elapsed')
    return df


def check_shares_sum(df_race2, df_race):

    # add up the shares in a file which contains votes, shares, per election
    def add_shares(df_race2):
        df_race2['Share'] = df_race2['Share'].astype(float)
        df = df_race2.groupby(['RaceID'])['Share'].sum().reset_index()
        df['Index'] = range(df.shape[0])
        for x in [10, 50, 90, 98, 101, 1000]:
            print "<", x, len(df[(df['Share'] < x)])
        df_race2_wrong_shares = df[df['Share'] < 50]
        return df_race2_wrong_shares

    # return a full list of incorrect races
    def shares_wrong_big(df_race2_wrong_shares, df_race, df_race2):
        # df is part of df_race2 with incorrect shares
        df_race2_wrong_shares['RaceID'] = df_race2_wrong_shares['RaceID'].astype(int)
        df_race['RaceID'] = df_race['RaceID'].astype(int)
        df = df_race.merge(df_race2_wrong_shares, left_on='RaceID', right_on='RaceID', how='outer')
        df_wrong_shares_full = df_race2.merge(df, left_on='RaceID', right_on='RaceID', how='outer')
        df_wrong_shares_full['Share_y'] = df_wrong_shares_full['Share_y'].astype(str)
        df_wrong_shares_full = df_wrong_shares_full[df_wrong_shares_full['Share_y'].str.contains(r'\d+')]
        df_wrong_shares_full.to_csv("pdata/wrong_shares_full.csv")
        return df_wrong_shares_full

    # return a list of RaceIDs for incorrect races
    def shares_wrong_small(df_wrong_shares_full):
        g = df_wrong_shares_full['RaceID'].unique()
        s = pd.Series(g)
        s.to_csv('pdata/wrong_shares_raceid.csv')

    df_race2_wrong_shares = add_shares(df_race2)
    df_wrong_shares_full = shares_wrong_big(df_race2_wrong_shares,df_race,df_race2)
    shares_wrong_small(df_wrong_shares_full)
    return df_wrong_shares_full

def unique_candidates(df_race2):
    g = df_race2['CandID'].unique()
    df_unique_CandID = pd.DataFrame(pd.Series(g)).rename(columns = {0:'CandID'})
    print 'number of unique candidates=', len(df_unique_CandID)
    df_unique_CandID.to_csv('pdata/unique_CandID.csv')
    return df_unique_CandID

def cand_remove(df, list):
    for x in list:
        df = df[df['CandID']!= x]
    return df

def recent_elections_city(folder):
    path = r'{}'.format(folder)
    df_m = pd.DataFrame()
    for id in [1,2,3]:
        df = pd.read_csv(path + "/recent_elections_part{}.txt".format(id), delimiter = ';', header = None)
        df_m = df_m.append(df)
    h = df_m.shape[0]
    df_m['CityID'] = range(h)
    dic = {0: 'web', 1: 'city', 2: "state", 3: 'partisan', 4: 'note'}
    for key, value in dic.iteritems():
        df_m = df_m.rename(columns={key: value})
    df_m.to_csv('pdata/recent_elections_city.csv')
    df_m = df_m.drop('note', 1)
    return df_m

def recent_elections_state(folder):
    path = r'{}'.format(folder)
    df_m = pd.DataFrame()
    df_m = pd.read_csv(path + "/governor.csv".format(id), delimiter = ',', header = None)
    df_m = df_m[~df_m[0].isnull()]
    h = df_m.shape[0]
    df_m['StateID'] = range(h)
    dic = {0: 'ContainerID', 1: 'State', 2: "year", 3: 'note'}
    for key, value in dic.iteritems():
        df_m = df_m.rename(columns={key: value})
    df_m.to_csv('pdata/recent_elections_state.csv')
    df_m = df_m.drop('note', 1)
    return df_m

def recent_elections(dist,folder_city,folder_state):
    if dist == 'City':
        df_m = recent_elections_city(folder_city)
    else:
        df_m = recent_elections_state(folder_state)
    return df_m

def city_name_merge(df_recent, df_race):
    df_recent['RaceID'] = df_recent['web'].str.extract('(\d+)', expand = False).astype(str)
    df_race['RaceID'] = df_race['RaceID'].astype(str)
    df = df_recent.merge(df_race, left_on = 'RaceID', right_on = 'RaceID', how = 'outer')
    df_city = df[['State', 'County', 'City', 'web', 'state', 'city', 'CityID']]
    df_city.loc[:,'CityID'] = df_city['CityID'].astype(str)
    df_city = df_city[df_city['CityID'].str.contains('(\d+)')]
    df_city = df_city[df_city['web'].str.contains('http')]
    df_city.to_csv('pdata/city_name_merge.csv')
    return df_city

def state_name_merge(df_recent, df_race):
    df_recent.loc[:, 'State'] = df_recent['State'].str.lower().str.strip()
    df_race.loc[:,'State'] = df_race['State'].str.lower().str.strip()
    df_state = df_recent.merge(df_race, left_on='State', right_on='State', how = 'outer')
    df_state = df_state[['State','StateID']]
    df_state.to_csv('pdata/state_name_merge.csv')
    return df_state

def dist_name_merge(df_recent, df_race, dist):
    if dist == 'City':
        df_dist = city_name_merge(df_recent, df_race)
    else:
        df_dist = state_name_merge(df_recent, df_race)
    return df_dist
