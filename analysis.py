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


def check_shares_sum():

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
        df_wrong_shares_full.to_csv("wrong_shares_full.csv")
        return df_wrong_shares_full

    # return a list of RaceIDs for incorrect races
    def shares_wrong_small(df_wrong_shares_full):
        g = df_wrong_shares_full['RaceID'].unique()
        s = pd.Series(g)
        s.to_csv('wrong_shares_raceid.csv')

    df_race2_wrong_shares = add_shares(df_race2)
    df_wrong_shares_full = shares_wrong_big(df_race2_wrong_shares,df_race,df_race2)
    shares_wrong_small(df_wrong_shares_full)
    return df_wrong_shares_full

def unique_candidates(df_race2):
    g = df_race2['CandID'].unique()
    df_unique_CandID = pd.DataFrame(pd.Series(g)).rename(columns = {0:'CandID'})
    print 'number of unique candidates=', len(df_unique_CandID)
    df_unique_CandID.to_csv('unique_CandID.csv')
    return df_unique_CandID

def cand_remove(df, list):
    for x in list:
        df = df[df['CandID']!= x]
    return df

def recent_elections_city():
    path = r'{}'.format(dir6)
    df_m = pd.DataFrame()
    for id in [1,2,3]:
        df = pd.read_csv(path + "/recent_elections_part{}.txt".format(id), delimiter = ';', header = None)
        df_m = df_m.append(df)
    h = df_m.shape[0]
    df_m['CityID'] = range(h)
    dic = {0: 'web', 1: 'city', 2: "state", 3: 'partisan', 4: 'note'}
    for key, value in dic.iteritems():
        df_m = df_m.rename(columns={key: value})
    df_m.to_csv('recent_elections_city.csv')
    df_m = df_m.drop('note', 1)
    return df_m

def recent_elections_state():
    path = r'{}'.format(dir7)
    df_m = pd.DataFrame()
    df_m = pd.read_csv(path + "/governor.csv".format(id), delimiter = ',', header = None)
    df_m = df_m[~df_m[0].isnull()]
    h = df_m.shape[0]
    df_m['StateID'] = range(h)
    dic = {0: 'ContainerID', 1: 'State', 2: "year", 3: 'note'}
    for key, value in dic.iteritems():
        df_m = df_m.rename(columns={key: value})
    df_m.to_csv('recent_elections_state.csv')
    df_m = df_m.drop('note', 1)
    return df_m

def recent_elections(dist):
    if dist == 'City':
        df_m = recent_elections_city()
    else:
        df_m = recent_elections_state()
    return df_m

def city_name_merge(df_recent, df_race):
    df_recent['RaceID'] = df_recent['web'].str.extract('(\d+)', expand = False).astype(str)
    df_race['RaceID'] = df_race['RaceID'].astype(str)
    df = df_recent.merge(df_race, left_on = 'RaceID', right_on = 'RaceID', how = 'outer')
    df_city = df[['State', 'County', 'City', 'web', 'state', 'city', 'CityID']]
    df_city.loc[:,'CityID'] = df_city['CityID'].astype(str)
    df_city = df_city[df_city['CityID'].str.contains('(\d+)')]
    df_city = df_city[df_city['web'].str.contains('http')]
    df_city.to_csv('city_name_merge.csv')
    return df_city

def state_name_merge(df_recent, df_race):
    df_recent.loc[:, 'State'] = df_recent['State'].str.lower().str.strip()
    df_race.loc[:,'State'] = df_race['State'].str.lower().str.strip()
    df_state = df_recent.merge(df_race, left_on='State', right_on='State', how = 'outer')
    df_state = df_state[['State','StateID']]
    df_state.to_csv('state_name_merge.csv')
    return df_state

def dist_name_merge(df_recent, df_race, dist):
    if dist == 'City':
        df_dist = city_name_merge(df_recent, df_race)
    else:
        df_dist = state_name_merge(df_recent, df_race)
    return df_dist

def race_details_recent(df_race, df_dist, distID, label):
    df_race_all = df_race.merge(df_dist, left_on = label, right_on = label, how = 'outer')
    df_race_distID = df_race_all.groupby([distID])['RaceID'].count().reset_index()
    print df_race_distID['RaceID'].describe()
    return df_race_all

def race_details2_recent(df_non_writein, df_race_all, distID):
    df_non_writein.loc[:, 'RaceID'] = df_non_writein['RaceID'].astype(float).astype(str)
    df_race_all.loc[:, 'RaceID'] = df_race_all['RaceID'].astype(float).astype(str)
    df_race2_all = df_non_writein.merge(df_race_all, left_on = ['RaceID'], right_on = ['RaceID'], how = 'outer')
    df_race2_distID = df_race2_all.groupby([distID])['CandID'].count().reset_index()
    print df_race2_distID['CandID'].describe()
    return df_race2_all

def terminal_election(df_race2_all, distID):
    df_terminal = df_race2_all.groupby([distID, 'Term Start Year'])['Polls Close_date'].max().reset_index()\
                  .rename(columns={'Polls Close_date': 'Terminal Date'})
    df_race2_all = df_race2_all.merge(df_terminal, left_on=[distID, 'Term Start Year'],right_on=[distID, 'Term Start Year'], how='outer')
    df_race2_all['Terminal'] = (df_race2_all['Polls Close_date'] == df_race2_all['Terminal Date'])
    return df_race2_all

def early_dist(df_race2_all, distID):
    df_early = df_race2_all.groupby([distID])['Term Start Year'].min().reset_index()\
               .rename(columns={'Term Start Year': 'Earlist Date'})
    df_race2_all = df_race2_all.merge(df_early, left_on=[distID], right_on=[distID], how='outer')
    df_race2_all['Earlist'] = (df_race2_all['Term Start Year'] == df_race2_all['Earlist Date'])
    return df_race2_all

def winner_election_period(df_race2_all, distID):
    df_winner = df_race2_all[df_race2_all['Terminal'] == True]
    df_winner = df_winner.groupby([distID, 'Term Start Year'])['Votes'].max().reset_index()\
                .rename(columns={'Votes': 'Votes Max Election Period'})
    df_race2_all = df_race2_all.merge(df_winner, left_on=[distID, 'Term Start Year'],right_on=[distID, 'Term Start Year'], how='outer')
    df_race2_all['winner'] = (df_race2_all['Votes'] == df_race2_all['Votes Max Election Period']) * 1.0
    df_race2_all['winnerID'] = df_race2_all['winner'] * df_race2_all['CandID'].astype(float)
    return df_race2_all

def follower_election_period(df_race2_all, distID):
    df_follower = df_race2_all[(df_race2_all['Terminal'] == True) & (df_race2_all['winner'] == False)]
    df_follower = df_follower.groupby([distID,'Term Start Year'])['Votes'].max().reset_index()\
                  .rename(columns = {'Votes':'Votes 2nd Election Period'})
    df_follower = df_follower[df_follower['Votes 2nd Election Period'].str.contains(r'\d+')]
    df_race2_all = df_race2_all.merge(df_follower, left_on=[distID, 'Term Start Year'], right_on=[distID, 'Term Start Year'], how='outer')
    df_race2_all['follower'] = (df_race2_all['Votes']==df_race2_all['Votes 2nd Election Period']) * 1.0
    df_race2_all['followerID'] = df_race2_all['follower'] * df_race2_all['CandID'].astype(float)
    return df_race2_all

def key_election(df_race2_all, distID):
    df_race2_all.loc[:,'Votes']=df_race2_all['Votes'].str.replace(",",'').astype(float)
    df_key1 = df_race2_all.groupby([distID,'Term Start Year','RaceID'])['Votes'].sum().reset_index()
    df_key2 = df_key1.groupby([distID, 'Term Start Year'])['Votes'].max().reset_index().rename(columns={'Votes':'Votes Sum Max'})
    df_key3 = df_key1.merge(df_key2,left_on=[distID,'Term Start Year'],right_on=[distID,'Term Start Year'],how='outer')
    df_key3['KeyRace'] = (df_key3['Votes']==df_key3['Votes Sum Max'])
    df_key3['KeyRaceID'] = (df_key3['Votes']==df_key3['Votes Sum Max'])*1.0*df_key3['RaceID'].astype(float)
    df_key3 = df_key3[['KeyRaceID','Votes Sum Max','KeyRace','RaceID']]
    df_race2_all = df_race2_all.merge(df_key3,left_on='RaceID',right_on='RaceID',how='outer')
    return df_race2_all

def winner_key_election(df_race2_all, distID):
    df_winner = df_race2_all[df_race2_all['KeyRace'] == True]
    df_winner = df_winner.groupby([distID, 'Term Start Year'])['Votes'].max().reset_index()\
                .rename(columns={'Votes': 'Votes Max Key Race'})
    df_race2_all = df_race2_all.merge(df_winner, left_on=[distID, 'Term Start Year'],right_on=[distID, 'Term Start Year'], how='outer')
    df_race2_all['winner_key'] = (df_race2_all['Votes'] == df_race2_all['Votes Max Key Race']) * 1.0
    df_race2_all['winner_keyID'] = df_race2_all['winner_key'] * df_race2_all['CandID'].astype(float)
    return df_race2_all

def follower_key_election(df_race2_all, distID):
    df_follower = df_race2_all[(df_race2_all['KeyRace'] == True) & (df_race2_all['winner_key'] == False)]
    df_follower = df_follower.groupby([distID,'Term Start Year'])['Votes'].max().reset_index()\
                  .rename(columns = {'Votes':'Votes 2nd Key Race'})
    df_follower.loc[:,'Votes 2nd Key Race'] = df_follower['Votes 2nd Key Race'].astype(str)
    df_follower = df_follower[df_follower['Votes 2nd Key Race'].str.contains(r'\d+')]
    df_race2_all = df_race2_all.merge(df_follower, left_on=[distID, 'Term Start Year'], right_on=[distID, 'Term Start Year'], how='outer')
    df_race2_all['follower_key'] = (df_race2_all['Votes']==df_race2_all['Votes 2nd Key Race'])*1.0
    df_race2_all['follower_keyID'] = df_race2_all['follower_key'] * df_race2_all['CandID'].astype(float)
    return df_race2_all

def win_follow_ever(df_race2_all):
    df = df_race2_all.groupby(['CandID'])['winner','follower','winner_key','follower_key'].max().reset_index()\
         .rename(columns={'winner':'winner ever','follower':'follower ever','winner_key':'winner_key ever','follower_key':'follower_key ever'})
    df_race2_all = df_race2_all.merge(df,left_on='CandID',right_on='CandID',how='outer')
    return df_race2_all

def incumbent_election_v1(df_race2_all, distID):
    df_race2_all['Name Flag'] = (df_race2_all['Name'].str.contains('I')) * 1.0
    df = df_race2_all.groupby([distID, 'Term Start Year'])['Name Flag'].sum().reset_index()
    df['Incumbent1'] = (df['Name Flag'] > 0.0) * 1.0
    df_race2_all = df_race2_all.merge(df, left_on=[distID, 'Term Start Year'], right_on=[distID, 'Term Start Year'], how='outer')
    return df_race2_all

def incumbent_election_v2(df_race2_all, distID):
    df_winner_id = df_race2_all[['winnerID', distID, 'Term Start Year']]\
                   .groupby([distID, 'Term Start Year'])['winnerID'].max().reset_index()\
                   .sort_values([distID, 'Term Start Year'], ascending=True)
    df_winner_id['winnerID previous'] = df_winner_id.groupby([distID])['winnerID'].shift(1)
    df_winner_id = df_winner_id.drop('winnerID', 1)

    df_race2_all = df_race2_all.merge(df_winner_id, left_on=[distID, 'Term Start Year'],
                                      right_on=[distID, 'Term Start Year'], how='outer')
    df_race2_all['CandID'] = df_race2_all['CandID'].astype(float)
    df_race2_all['winnerID previous'] = df_race2_all['winnerID previous'].astype(float)
    df_race2_all['Matched'] = (df_race2_all['CandID'] == df_race2_all['winnerID previous']) * 1.0

    df = df_race2_all.groupby([distID,'Term Start Year'])['Matched'].max().reset_index()
    df = df[df['Matched'] > 0]
    df = df.rename(columns = {'Matched':'Incumbent2'})
    df_race2_all = df_race2_all.merge(df, left_on = [distID,'Term Start Year'], right_on = [distID, 'Term Start Year'], how = 'outer')
    df_race2_all.loc[df_race2_all['Incumbent2'] != 1,'Incumbent2'] = (df_race2_all['Earlist'])* 2.0
    # Incumbent2 = {1:incumbent, 0:open, 2:unclear}

    return df_race2_all

def career_span(df_race2_all):
    df1 = df_race2_all.groupby(['CandID'])['Term Start Year'].agg([min, max]).reset_index().rename(columns={'min':'Career Start Year','max':'Career End Year'})
    df_race2_all = df_race2_all.merge(df1, left_on='CandID', right_on='CandID',how='outer')
    return df_race2_all

def sam(df_race2_all):
    df_sam0 = pd.read_excel('Mayoral_candidate_bios.xlsx')
    print df_sam0['CandID'].nunique()
    df_sam = df_sam0.groupby(['CandID'])['name'].count().reset_index().rename(columns={'name':'Sam'})
    df_sam = df_sam[~ df_sam['CandID'].isnull()]

    df_race2_all = df_race2_all.merge(df_sam, left_on='CandID', right_on='CandID', how ='outer')
    df_race2_all.loc[:,'Sam'] = df_race2_all['Sam'].astype(str)
    df_race2_all.loc[:,'Sam'] = (df_race2_all['Sam'].str.contains('(\d+)'))*1.0
    return df_race2_all

def sam_source(df_race2_all):
    df_race2_all['Sam1'] = df_race2_all['Sam2'] = df_race2_all['Sam3'] = df_race2_all['Sam4'] = df_race2_all['Sam5']=''

    dic_wiki = {'3461':'http://en.wikipedia.org/wiki/Kirk_Watson',
           '9396':'http://en.wikipedia.org/wiki/Carl_Stokes_(Baltimore)',
           '17346':'http://en.wikipedia.org/wiki/Catherine_E._Pugh',
           '21736':'http://en.wikipedia.org/wiki/Keiffer_J._Mitchell,_Jr.',
           '8973':'http://en.wikipedia.org/wiki/Kurt_Schmoke',
           '6757':"http://en.wikipedia.org/wiki/Martin_O'Malley'",
           '21742':'http://en.wikipedia.org/wiki/Mary_Pat_Clarke',
           '17344':'http://en.wikipedia.org/wiki/Sheila_Dixon#Career',
           '21729':'http://en.wikipedia.org/wiki/Stephanie_Rawlings-Blake',
           '138395':'http://en.wikipedia.org/wiki/Michael_F._Flaherty',
           '12593':'http://en.wikipedia.org/wiki/Raymond_Flynn',
           '28699':'http://en.wikipedia.org/wiki/Thomas_Menino',
           '100794':'http://en.wikipedia.org/wiki/Anthony_Foxx',
           '11001':'http://en.wikipedia.org/wiki/Pat_McCrory',
           '123':'http://en.wikipedia.org/wiki/Richard_Vinroot',
           '10867':'http://en.wikipedia.org/wiki/Michael_B._Coleman',
           '6236':'http://en.wikipedia.org/wiki/Laura_Miller',
           '372419':'https://en.wikipedia.org/wiki/Ron_Kirk#Post-mayoral_career',
           '18567':'http://en.wikipedia.org/wiki/Steve_Bartlett',
           '135592':'http://en.wikipedia.org/wiki/Tom_Leppert',
           '198243':'http://en.wikipedia.org/wiki/Dave_Bing#College',
           '81655':'http://en.wikipedia.org/wiki/Freman_Hendrix',
           '3707':'http://en.wikipedia.org/wiki/Kwame_Kilpatrick',
           '95990':'http://en.wikipedia.org/wiki/John_Cook_(mayor_of_El_Paso)',
           '9518':'http://en.wikipedia.org/wiki/Bart_Peterson',
           '166708':'http://en.wikipedia.org/wiki/Greg_Ballard',
           '143913':'http://en.wikipedia.org/wiki/Alvin_Brown',
           '6986':'http://en.wikipedia.org/wiki/John_Delaney_(Florida_politician)',
           '6982':'http://en.wikipedia.org/wiki/Nat_Glover',
           '2331':'http://en.wikipedia.org/wiki/Carol_Chumney',
           '159097':'http://en.wikipedia.org/wiki/Richard_Hackett',
           '1712':'https://en.wikipedia.org/wiki/Bob_Clement',
           '135612':'http://en.wikipedia.org/wiki/Karl_Dean',
           '19646':'http://en.wikipedia.org/wiki/Greg_Nickels#Political_career',
           '111116':'http://en.wikipedia.org/wiki/Cindy_Chavez',
           '111112':'http://en.wikipedia.org/wiki/Chuck_Reed',
           '19414':'http://en.wikipedia.org/wiki/Willie_Brown_(politician)#After_Mayorship',
           '8727':'http://en.wikipedia.org/wiki/Tom_Ammiano',
           '201052':'http://en.wikipedia.org/wiki/John_Avalos',
           '8728':'http://en.wikipedia.org/wiki/Gavin_Newsom',
           '19416':'http://en.wikipedia.org/wiki/Frank_Jordan#Personal',
           '261622':'http://en.wikipedia.org/wiki/Ed_Lee_(politician)',
           '25739':'http://en.wikipedia.org/wiki/Art_Agnos',
           '13979':'http://en.wikipedia.org/wiki/Susan_Golding',
           '88601':'http://en.wikipedia.org/wiki/Steve_Francis_(businessman)',
           '87396':'http://en.wikipedia.org/wiki/Jerry_Sanders_(politician)',
           '61131':'http://en.wikipedia.org/wiki/Donna_Frye',
           '13967':'http://en.wikipedia.org/wiki/Dick_Murphy',
           '92315':'http://en.wikipedia.org/wiki/Ron_Gonzales',
           '215061':'http://en.wikipedia.org/wiki/Michael_McGinn',
           '8608':'http://en.wikipedia.org/wiki/Norm_Rice',
           '19648':'http://en.wikipedia.org/wiki/Paul_Schell',
           '54581':'http://en.wikipedia.org/wiki/Adrian_Fenty',
           '3989':'http://en.wikipedia.org/wiki/Carol_Schwartz',
           '54564':'https://en.wikipedia.org/wiki/Linda_W._Cropp',
           '26431':'http://en.wikipedia.org/wiki/Marion_Barry#Early_life.2C_education.2C_and_civil_rights_activism',
           '32070':'http://en.wikipedia.org/wiki/Sharon_Pratt_Kelly',
           '60497':'http://en.wikipedia.org/wiki/Vincent_C._Gray',
           '86792':'http://en.wikipedia.org/wiki/Phil_Gordon_(politician)',
           '105991':'http://en.wikipedia.org/wiki/Tom_Knox',
           '4573':'https://en.wikipedia.org/wiki/Sam_Katz_(Philadelphia)',
           '47026':'http://en.wikipedia.org/wiki/Michael_Nutter',
           '43012':'http://en.wikipedia.org/wiki/Lucien_E._Blackwell',
           '4572':'http://en.wikipedia.org/wiki/John_F._Street',
           '16':'http://en.wikipedia.org/wiki/Ed_Rendell#Personal_life',
    }
    dic_linkedin = {'6236':'http://www.linkedin.com/pub/laura-miller/1b/a82/546',
                    '135592':'http://www.linkedin.com/in/tomleppert',
                    '92315':'http://www.linkedin.com/pub/ron-gonzales/5/b45/106',
                    '13979':'http://www.linkedin.com/pub/susan-golding/27/58a/9a1',
                    '86792':'http://www.linkedin.com/pub/phil-gordon/38/6a4/257',
                    '105991':'http://www.linkedin.com/pub/thomas-knox/a/682/5b5',
                    '15921':'http://www.linkedin.com/pub/al-taubenberger/23/482/a33',
                    '4597':'http://www.linkedin.com/in/rudygiuliani',
                    }

    dic_others =   {'1075':['http://rush.house.gov/'],
                    '6427':['http://ajwright.com/documents/Rev.PaulJakesCV.pdf'],
                    '20022':['http://library.csu.edu/collections/pincham/history/'],
                    '5894':['http://votesmart.org/candidate/biography/8018/sylvester-turner#.UelFN2T72XQ','http://ballotpedia.org/wiki/index.php/Sylvester_Turner'],
                    '8270':['http://www.nytimes.com/2005/08/10/nyregion/10biobox.html?_r=0','http://iarchives.nysed.gov/xtf/view?docId=03-01_03-04a_03-04b_03-05_04-53.xml;query=;brand=default'],
                    '8727':['http://www.asmdc.org/members/a17/'],
                    '47026':['http://www.thehistorymakers.com/biography/hon-michael-nutter'],
                    '16':['http://www.ballardspahr.com/eventsnews/pressreleases/2011-01-24_edwardrendellreturnstoballardspahr.aspx'],
                    '15921':['https://cityroom.blogs.nytimes.com/2009/05/27/emboldened-thompson-presses-his-mayoral-bid/?scp=4&sq=2009%20mayor%20race&st=cse'],
                    '8073':['http://web.archive.org/web/20010731182537/','http://www.nypn.org/htm/resources/ruth-messinger.html','http://web.archive.org/web/19970121104613/','http://ruth97.org/',],
                    '4597':['http://www.biography.com/people/rudolph-giuliani-9312674','http://www.nyc.gov/html/records/rwg/html/bio.html'],
                    '4634':['http://www.nyc.gov/portal/site/nycgov/menuitem.e985cf5219821bc3f7393cd401c789a0','http://www.businessinsider.com/michael-bloomberg-biography-2012-7?op=1','http://www.mikebloomberg.com/index.cfm?objectid=e689d66f-96fd-e9f6-b1af64b8dae78a69'],
                    '6623':['http://www.huffingtonpost.com/author/mark-green']
                  }

    #http://en.wikipedia.org/wiki/Walter_Moore_(politician)
    #https://en.wikipedia.org/wiki/Greg_Lashutka
    #http://en.wikipedia.org/wiki/Bob_Lanier_(politician)
    #http://www.houstontx.gov/mayor/bio.html
    #http://en.wikipedia.org/wiki/Matt_Gonzalez#2008_presidential_race
    #'2813':

    df_race2_all['Wikipedia'] = df_race2_all['Linkedin'] = df_race2_all['Others1'] = df_race2_all['Others2'] = df_race2_all['Others3'] = df_race2_all['Others4'] = ""
    df_race2_all['CandIDs'] = df_race2_all['CandID'].astype(float)
    for key, value in dic_wiki.iteritems():
        df_race2_all.loc[df_race2_all['CandIDs']==float(key),'Wikipedia']=value
    for key, value in dic_linkedin.iteritems():
        df_race2_all.loc[df_race2_all['CandIDs']==float(key),'Linkedin']=value
    for key, value in dic_others.iteritems():
        h = len(value)
        for i in range(h):
            df_race2_all.loc[df_race2_all['CandIDs']==float(key),'Others{}'.format(i+1)] = value[i]
    return df_race2_all

def statistics_dist(df_recent, df_dist, df_periods, df_race_all, dist, dists, distID):
    stat_dist = dict()

    s = len(df_recent[dist])
    print 'Total {}'.format(dists), s
    stat_dist['Total {}'.format(dists)] = s

    if dists == 'city':
       s = len(df_recent[df_recent['web'].str.contains('http')])
       print 'Total {} with Data'.format(dists), s
       stat_dist['Total {} with Data'.format(dists)] = s

    df_dist[distID] = df_dist[distID].astype(float)
    s = df_dist[distID].mean()
    print 'Avg Ranks', s
    stat_dist['avg Ranks'] = s

    s = df_dist[distID].median()
    print 'Median Ranks', s
    stat_dist['Median Ranks'] = s

    df_periods_dist = df_periods.groupby([distID])['Term Start Year'].count().reset_index()
    s = df_periods_dist['Term Start Year'].mean()
    print 'Avg Election Periods by {}:'.format(dist), s
    stat_dist['Avg Election Periods'] = s

    df_elections_dist = df_race_all.groupby([distID])['RaceID'].count().reset_index()
    s = df_elections_dist['RaceID'].mean()
    print 'Avg Elections by {}:'.format(dist), s
    stat_dist['Avg Elections'] = s

    df_term_dist = df_race_all.groupby([distID])['Term Length'].mean().reset_index()
    s = df_term_dist['Term Length'].mean()
    print 'Avg Term Length:', s
    stat_dist['Avg Term Lengths'] = s

    return stat_dist

def statistics_election(df_periods, df_race2_all, distID):
    stat_election = dict()

    s = df_periods['RaceID'].sum()
    print 'Elections Covered', s
    stat_election['Election Covered'] = s

    s = df_periods['RaceID'].count()
    print 'Number of Election Periods', s
    stat_election['Election Periods Covered'] = s

    df = df_race2_all[df_race2_all['Incumbent2'] == 1]
    df = df.groupby([distID,'Term Start Year'])['RaceID'].max().reset_index()
    s = df['Term Start Year'].count()
    print 'Number of Incumbent Election Periods', s
    stat_election['Incumbent Election Periods'] = s

    df2 = df_race2_all[df_race2_all['Incumbent2'] == 1]
    g = df2['RaceID'].unique()
    s = pd.Series(g).count()
    print 'Number of Unique Incumbent Candidates', s
    stat_election['Incumbent Election Candidates'] = s

    df = df_race2_all[df_race2_all['Incumbent2'] == 0]
    df = df.groupby([distID, 'Term Start Year'])['RaceID'].max().reset_index()
    s = df['Term Start Year'].count()
    print 'Number of Open Election Periods', s
    stat_election['Open Election Periods'] = s

    df2 = df_race2_all[df_race2_all['Incumbent2'] == 0]
    g = df2['RaceID'].unique()
    s = pd.Series(g).count()
    print 'Number of Unique Open Candidates', s
    stat_election['Open Election Candidates'] = s

    df = df_race2_all[df_race2_all['Incumbent2'] == 2]
    df = df.groupby([distID, 'Term Start Year'])['RaceID'].max().reset_index()
    s = df['Term Start Year'].count()
    print 'Number of Unclear Election Periods', s
    stat_election['Unclear Election Periods'] = s

    df2 = df_race2_all[df_race2_all['Incumbent2'] == 2]
    g = df2['RaceID'].unique()
    s = pd.Series(g).count()
    print 'Number of Unique Unclear Candidates', s
    stat_election['Unclear Election Candidates'] = s

    return stat_election

def select_districts(df_race2_all, key1, cutoff1, key2, cutoff2):
    df_race2_all.loc[:, key1] = df_race2_all[key1].astype(float)
    print len(df_race2_all)
    df_race2_all = df_race2_all[df_race2_all[key1] < cutoff1]
    df_race2_all = df_race2_all[df_race2_all[key2] > cutoff2]
    print len(df_race2_all)
    df_non_writein_id = df_race2_all.groupby(['CandID'])['RaceID'].count().reset_index().rename(columns={'RaceID': 'RaceIDs'})
    return df_race2_all, df_non_writein_id

def statistics_candidates():
    def cand_inc_cha_open(df_race2_all, df_non_writein_id):
        df_race_ct = df_non_writein_id
        dic0 = {'elections': 'RaceID', 'election periods': 'Term Start Year'}
        dic1 = {'Open': df_race2_all['Incumbent2'] == 0,
                'Incumbent': (df_race2_all['Incumbent2'] == 1) & (
                    df_race2_all['CandID'] == df_race2_all['winnerID previous']),
                'Challenger': (df_race2_all['Incumbent2'] == 1) & (
                    df_race2_all['CandID'] != df_race2_all['winnerID previous']),
                'Unclear': df_race2_all['Incumbent2'] == 2}
        for label0, value0 in dic0.iteritems():
            for label1, value1 in dic1.iteritems():
                df = df_race2_all[value1].groupby(['CandID'])[value0].nunique().reset_index().rename(
                     columns={value0: '{} {}'.format(label1, label0)})
                df_non_writein_id.loc[:, 'CandID'] = df_non_writein_id['CandID'].astype(float).astype(str)
                df.loc[:, 'CandID'] = df['CandID'].astype(float).astype(str)
                df2 = df_non_writein_id.merge(df, left_on='CandID', right_on='CandID', how='outer')
                df2.loc[df2['{} {}'.format(label1, label0)].isnull(), '{} {}'.format(label1, label0)] = 0
                s = df2['{} {}'.format(label1, label0)].mean()
                stat_cand['Avg number of {} {}'.format(label1, label0)] = s
                df_race_ct = df_race_ct.merge(df2, left_on='CandID', right_on='CandID', how='outer')
        return ()

    def win_lose(df_race2_all):
        df_winner_list = df_race2_all.groupby(['winnerID'])['Term Start Year'].nunique().reset_index().rename(
            columns={'winnerID': 'CandID', 'Term Start Year': 'Win at once (Wins)'})
        df_winner_list = df_winner_list[df_winner_list['CandID'] > 0.0]
        s1 = len(df_winner_list)
        s2 = df_winner_list['Win at once (Wins)'].mean()

        df_race2_all_winner = df_race2_all.merge(df_winner_list, left_on='CandID', right_on='CandID', how='right')
        df = df_race2_all_winner.groupby(['CandID'])['Term Start Year'].nunique().reset_index()
        s3 = df['Term Start Year'].mean()

        df = df_race2_all.merge(df_winner_list, left_on='CandID', right_on='CandID', how='outer')
        df_race2_all_loser = df[df['Win at once (Wins)'].isnull()]
        df_loser_list = df_race2_all_loser.groupby(['CandID'])['Term Start Year'].nunique().reset_index()
        s4 = len(df_loser_list)
        s5 = df_loser_list['Term Start Year'].mean()

        df = df_race2_all_winner[df_race2_all_winner['CandID'] == df_race2_all_winner['winnerID']].groupby(['CandID'])[
            'Term Start Year'].min().reset_index() \
            .rename(columns={'Term Start Year': 'First Win Year'})
        df_race2_all_winner_1st = df_race2_all_winner.merge(df, left_on='CandID', right_on='CandID', how='outer')

        return s1,s2,s3,s4,s5,df_winner_list, df_race2_all_winner, df_race2_all_loser, df_race2_all_winner_1st

    def win_once_early_fails(df_race2_all_winner_1st, df_winner_list):
        df = df_race2_all_winner_1st[df_race2_all_winner_1st['Term Start Year'] < df_race2_all_winner_1st['First Win Year']]
        df = df.groupby(['CandID'])['Term Start Year'].nunique().reset_index().rename(
            columns={'Term Start Year': 'Win at once (Early Fails)'})
        df = df_winner_list.merge(df, left_on='CandID', right_on='CandID', how='left')
        df.loc[df['Win at once (Early Fails)'].isnull(), 'Win at once (Early Fails)'] = 0.0
        s = df['Win at once (Early Fails)'].mean()
        return s

    def win_once_late_tries(df_race_late, df_winner_list):
        df = df_race_late.groupby(['CandID'])['Term Start Year'].nunique().reset_index().rename(
            columns={'Term Start Year': 'Win at once (Late Tries)'})
        df_late = df_winner_list.merge(df, left_on='CandID', right_on='CandID', how='left')
        df_late.loc[df_late['Win at once (Late Tries)'].isnull(), 'Win at once (Late Tries)'] = 0.0
        s = df_late['Win at once (Late Tries)'].mean()
        return s

    def win_once_late_wins(df_race_late, df_winner_list):
        df = df_race_late[df_race_late['CandID'] == df_race_late['winnerID']].groupby(['CandID'])[
            'Term Start Year'].nunique().reset_index().rename(columns={'Term Start Year': 'Win at once (Late Wins)'})
        df_late_win = df_winner_list.merge(df, left_on='CandID', right_on='CandID', how='left')
        df_late_win.loc[df_late_win['Win at once (Late Wins)'].isnull(), 'Win at once (Late Wins)'] = 0.0
        s = df_late_win['Win at once (Late Wins)'].mean()
        return s

    def win_once_late_fails(df_race_late, df_winner_list):
        df = df_race_late[df_race_late['CandID'] != df_race_late['winnerID']].groupby(['CandID'])[
            'Term Start Year'].nunique().reset_index().rename(columns={'Term Start Year': 'Win at once (Late Fails)'})
        df_late_win = df_winner_list.merge(df, left_on='CandID', right_on='CandID', how='left')
        df_late_win.loc[df_late_win['Win at once (Late Fails)'].isnull(), 'Win at once (Late Fails)'] = 0.0
        s = df_late_win['Win at once (Late Fails)'].mean()
        return s


    stat_cand = dict()

    s = df_non_writein_id['CandID'].nunique()
    print 'Number of Unique Candidates', s
    stat_cand['Number of Unique Candidates'] = len(df_non_writein_id)

    df = df_race2_all.groupby(['CandID'])['Term Start Year'].nunique().reset_index()
    s = df['Term Start Year'].mean()
    print 'Number of Election Periods Per Candidate', s
    stat_cand['Number of Election Periods Per Candidate'] = s

    cand_inc_cha_open(df_race2_all, df_non_writein_id)

    s1, s2, s3, s4, s5, df_winner_list, df_race2_all_winner, df_race2_all_loser, df_race2_all_winner_1st = win_lose(
    df_race2_all)
    print 'Number of Candidates at least winning once', s1
    stat_cand['Number of Candidates at least winning once'] = s1
    print 'Winners: Number of Winning Election Periods', s2
    stat_cand['Winners: Number of Winning Election Periods'] = s2
    print 'Winners: Number of Election Periods', s3
    stat_cand['Winners: Number of Election Periods'] = s3
    print 'Number of Candidates never win', s4
    stat_cand['Number of Candidates never win'] = s4
    print 'Losers: Number of Election Periods', s5
    stat_cand['Losers: Number of Election Periods'] = s5

    s = win_once_early_fails(df_race2_all_winner_1st, df_winner_list)
    print 'Winners: Number of Failed Tries before First Win', s
    stat_cand['Winners: Number of Failed Tries before First Win'] = s

    df_race_late = df_race2_all_winner_1st[df_race2_all_winner_1st['Term Start Year'] > df_race2_all_winner_1st['First Win Year']]

    s = win_once_late_tries(df_race_late, df_winner_list)
    print 'Winners: Number of Tries After First Win', s
    stat_cand['Winners: Number of Tries After First Win'] = s

    s = win_once_late_wins(df_race_late, df_winner_list)
    print 'Winners: Number of Wins After First Win', s
    stat_cand['Winners: Number of Wins After First Win'] = s

    s = win_once_late_fails(df_race_late, df_winner_list)
    print 'Winners: Number of Fails After First Win', s
    stat_cand['Winners: Number of Fails After First Win'] = s

    return stat_cand

if __name__ == '__main__':
    # ====================================================== #
    #    Initialize Directory and load Data                  #
    # ====================================================== #

    dir0 = '/Users/yuwang/Documents/research/research/timing/git'
    dir1 = dir0 + '/analysis'
    dir2 = dir0 + '/campaigns/data'
    dir3 = dir0 + '/mayors/data'
    dir4 = dir0 + '/campaigns/schema'
    dir5 = dir0 + '/mayors/schema'
    dir6 = dir0 + '/mayors'
    dir7 = dir0 + '/campaigns'


    # create a folder for cache
    if not os.path.exists('pdata'):
        os.mkdir('pdata')
    if os.path.exists('key_race_details.csv'):
        os.remove('key_race_details.csv')
    if os.path.exists('key_race_details2.csv'):
        os.remove('key_race_details2.csv')

    # Create a dictionary for governor and mayor
    dicts = ['Mayor', 'City', 'CityID','city','Cities', ['State','City']]
    #dicts = ['Governor', 'State', 'StateID', 'State', 'States', 'State']

    key_race_details(dir3, dicts[0], 'key_race_details.csv')
    key_race_details2(dir3, dicts[0], 'key_race_details2.csv', 'Final Votes')

    df_race = setup_race_details(dicts[1])
    df_race2 = setup_race_details2()

    # ====================================================== #
    #    Data Cleaning                                       #
    # ====================================================== #
    # Check if all shares in a race add up to 100
    df_shares_wrong = check_shares_sum()

    # Generate a list of unique candidates
    df_unique_CandID = unique_candidates(df_race2)

    # Remove write-in candidates, until max number of mayoral elections per candidate is reasonable
    df_non_writein = cand_remove(df_race2, ['22593', '191', '19359', '30530','4666'])  # write-in & others ,'4667'
    df_non_writein_id = df_non_writein.groupby(['CandID'])['RaceID'].count().reset_index()
    df_non_writein_id = df_non_writein_id.sort_values(['RaceID'], ascending=True)
    print df_non_writein_id['RaceID'].describe()

    # Load the list of largest cities and merge the city names with those in ourcampaigns
    df_recent = recent_elections(dicts[1])
    df_dist = dist_name_merge(df_recent, df_race,dicts[1])

    # df_race_all is the master copy for race_details combined with recent elections
    df_race_all = race_details_recent(df_race, df_dist, dicts[2], dicts[5])

    # df_race2_all is the master copy for race_details, race_details2 combined with recent elections
    df_race2_all = race_details2_recent(df_non_writein, df_race_all, dicts[2])

    # df_periods is the master copy for [city, election periods]
    df_periods = df_race_all.groupby([dicts[2], 'Term Start Year'])['RaceID'].count().reset_index()

    # Mark the terminal election in each election period
    df_race2_all = terminal_election(df_race2_all, dicts[2])

    # Mark the earliest election period in each city
    df_race2_all = early_dist(df_race2_all,dicts[2])

    # Mark the winner in the terminal election in each election period
    df_race2_all = winner_election_period(df_race2_all,dicts[2])

    # Mark the follower (2nd place runner) in the terminal election in each election period
    df_race2_all = follower_election_period(df_race2_all,dicts[2])

    # Mark the key election in each election period
    df_race2_all = key_election(df_race2_all, dicts[2])

    # Mark the winner in the key election in each election period
    df_race2_all = winner_key_election(df_race2_all, dicts[2])

    # Mark the follower (2nd place runner) in the key election in each election period
    df_race2_all = follower_key_election(df_race2_all, dicts[2])

    # Mark the candidates who have ever made to mayor position
    df_race2_all = win_follow_ever(df_race2_all)

    # First way of differentiating incumbent/open elections: whether name contains '(I)'
    df_race2_all = incumbent_election_v1(df_race2_all, dicts[2])

    # Second way of differentiating incumbent/open elections: whether the winner of last period appears again
    df_race2_all = incumbent_election_v2(df_race2_all, dicts[2])

    # Mark the earliest and latest year of race for each candidate
    df_race2_all = career_span(df_race2_all)

    # Mark the candidate searched by Sam Gerson
    df_race2_all = sam(df_race2_all)
    df_race2_all = sam_source(df_race2_all)


    df_race2_all.to_csv('race2_all.csv')
    # ====================================================== #
    #     Summary Statistics for Cities                      #
    # ====================================================== #
    stat_dist = statistics_dist(df_recent, df_dist, df_periods, df_race_all, dicts[3],dicts[4],dicts[2])

    # ====================================================== #
    #    Summary Statistics for Elections                    #
    # ====================================================== #
    stat_election = statistics_election(df_periods, df_race2_all, dicts[2])

    # ====================================================== #
    #    Summary Statistics for Candidates                   #
    # ====================================================== #

    #df_race2_all, df_non_writein_id = select_districts(df_race2_all, 'CityID', 100, 'Term Start Year', 1950)
    stat_cand = statistics_candidates()

    # ====================================================== #
    #    List of Names for RA                                #
    # ====================================================== #
    df_name_RA = df_race2_all[['Name','City','CityID','CandID','Sam','winner ever','winner_key ever',
                               'follower ever','follower_key ever','Career Start Year','Career End Year',
                               'Wikipedia','Linkedin','Others1', 'Others2', 'Others3', 'Others4']]

    df_name_RA = df_name_RA.groupby(['City','CityID','CandID','Sam','winner ever','winner_key ever',
                                     'follower ever','follower_key ever','Career Start Year','Career End Year',
                                     'Wikipedia', 'Linkedin', 'Others1', 'Others2', 'Others3', 'Others4'])['Name'].min().reset_index()

    df_name_RA.loc[:, 'CityID'] = df_name_RA['CityID'].astype(float)
    df_name_RA = df_name_RA.sort_values(['CityID', 'CandID'], ascending=True)
    df_name_RA['Web'] = df_name_RA['CandID'].astype(int).astype(str)
    df_name_RA.loc[:, 'Web'] = 'http://www.ourcampaigns.com/CandidateDetail.html?CandidateID=' + df_name_RA['Web']

    print df_name_RA['CandID'].nunique()

    df_name_RA = df_name_RA[df_name_RA['winner ever']+df_name_RA['follower ever']+df_name_RA['winner_key ever']+df_name_RA['follower_key ever']>0]
    print df_name_RA['CandID'].nunique()

    df_name_RA = df_name_RA[df_name_RA['Career End Year']>1970]
    print df_name_RA['CandID'].nunique()

    df_name_RA.loc[:,'CityID'] = df_name_RA['CityID'].astype(float)
    df_name_RA.loc[df_name_RA['CityID']==21,'City'] = "DC"
    df_name_RA = df_name_RA.sort_values(['CityID','Career Start Year'],ascending=[True, False])
    df_name_RA = df_name_RA[df_name_RA['CityID']<150]
    print df_name_RA['CandID'].nunique()
    print 'sam marked:', df_name_RA['Sam'].sum()

    df_name_RA = df_name_RA[['Name','CandID','City','CityID','Wikipedia','Linkedin','Others1','Others2','Others3','Others4','Web']]
    df_name_RA = df_name_RA.reset_index().drop('index',1)
    df_name_RA.to_csv('name_RA.csv')

    #writer = pd.ExcelWriter('name_RA.xlsx')
    #df_name_RA.to_excel(writer, 'Sheet1')
    #writer.save()




