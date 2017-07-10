import os
import re
import csv
import json
import numpy
import pandas
from collections import defaultdict

def json2csv(count, file1, csvs):
    # file1 is a loaded json file
    file2 = open(csvs, 'a')
    csvwriter = csv.writer(file2)
    if count == 0: # write columns
       header = file1.keys()
       csvwriter.writerow(header)
       count += 1
    if count == 1: # skip columns
       csvwriter.writerow(file1.values())
    file2.close()

def key_race_details(folder, office, csvs):
    if os.path.exists(csvs):
        os.remove(csvs)
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



if __name__ == '__main__':
    # create a folder for cache
    if not os.path.exists('pdata'):
        os.mkdir('pdata')

    dir0 = '/Users/yuwang/Documents/research/research/timing/git'
    dir1 = '/Users/yuwang/Documents/research/research/timing/git/analysis'
    dir2 = '/Users/yuwang/Documents/research/research/timing/git/campaigns/data'
    dir3 = '/Users/yuwang/Documents/research/research/timing/git/mayors_test/data'
    dir4 = '/Users/yuwang/Documents/research/research/timing/git/campaigns/schema'
    dir5 = '/Users/yuwang/Documents/research/research/timing/git/mayors/schema'


    key_race_details(dir3,'Mayor', 'test.csv')
    #json2csv(0, "race_WorcesterMayor_1988_561866_RaceDetails.json", "test.csv")



'''
def keyrace(office, folder1, folder2):
    for file in os.listdir(folder2):
        filename = os.path.basename(file)
        if ("race_" in filename) & (office in filename) & ("RaceDetails" in filename):
            x = re.findall(r'\d+', str(filename))
            i = x[1]
            with open(os.path.join(folder2, file)) as f2:
                son = json.load(f2)
                son['RaceID'] = i
                print(son['RaceID'])
                raw_input("Enter")
                dd = defaultdict(list)
                for d in (ms, son):  # you can list as many input dicts as you want here
                    for key, value in d.iteritems():
                        dd[key].append(value)
                ms = dd
    return ms

def keyrace(office, folder):
    Iterate all the files in the folder
    if filename contains "race_", "Mayor", "RaceDetails"
        add to big dictionary key_office_Racedetails
    if filename contains "race_", "Mayor", "Candidates"
        add to big dictionary key_office_Racecandidates

def keycand(office, folder):
    if cand_id in filename is also contained in key_office_racecandidates
        add to big dictionary key_office_candidates
'''


