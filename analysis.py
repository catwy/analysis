import os
import re
import csv
import json
import numpy
import pandas
from collections import defaultdict

def keyrace(office, folder1, folder2):
    with open(os.path.join(folder1, 'race.json')) as f1:
        ms1 = json.load(f1)
        ms1['RaceID'] = ""

        for file in os.listdir(folder2):
            filename = os.path.basename(file)
            if ("race_" in filename) & (office in filename) & ("RaceDetails" in filename):
                x = re.findall(r'\d+', str(filename))
                i = x[1]

                with open(os.path.join(folder2, file)) as f2:
                    ms2 = json.load(f2)
                    ms2['RaceID'] = i

                    dd = defaultdict(list)
                    for d in (ms1, ms2):  # you can list as many input dicts as you want here
                        for key, value in d.iteritems():
                            dd[key].append(value)
                    ms1 = dd
                    for x in ms1:
                        print("mm",ms1['RaceID'])
                        raw_input("Enter")
    return ms1



if __name__ == '__main__':
    # create a folder for cache
    if not os.path.exists('pdata'):
        os.mkdir('pdata')

    dir0 = '/Users/yuwang/Documents/research/research/timing/git'
    dir1 = '/Users/yuwang/Documents/research/research/timing/git/analysis'
    dir2 = '/Users/yuwang/Documents/research/research/timing/git/campaigns/data'
    dir3 = '/Users/yuwang/Documents/research/research/timing/git/mayors/data'
    dir4 = '/Users/yuwang/Documents/research/research/timing/git/campaigns/schema'
    dir5 = '/Users/yuwang/Documents/research/research/timing/git/mayors/schema'

    keyrace("Mayor", dir5, dir3)

'''
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


