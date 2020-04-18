from pyedflib import highlevel
import numpy as np
import pandas as pd


def _time_sec(a):
    a = str(a[:10])
    a = a.replace(':','')
    if len(a) < 6:
        a = '0'+a
    res = 360*int(a[:2]) + 60*int(a[2:4]) + int(a[4:6])
    return res

loc = "../../data/stvinc/"

names = pd.ExcelFile(loc+"SubjectDetails.xls").parse(0)['Study Number'].to_numpy()
indexes = pd.ExcelFile(loc+"SubjectDetails.xls").parse(0)['S/No'].to_numpy()
start_time = pd.ExcelFile(loc+"SubjectDetails.xls").parse(0)['PSG Start Time'].to_numpy()

start_time = [_time_sec(a) for a in start_time]
names = [i.lower() for i in names]
indexes = [i-1 for i in indexes]


def parse_file(name):
    ind = name[0]
    name = name[1]
    signals, a, b = highlevel.read_edf(loc+name+'.rec')
    file1 = open(loc+name+'_respevt.txt',"r") 

    lines = file1.readlines()[3:-2]

    sound = signals[7]
    label = np.zeros(sound.shape)

    for l in lines:
        st = _time_sec(l) - start_time[ind]
        duration = int(l[28:30])
        label[st*8:st*8+duration*8].fill(1)

    return sound, label

def get_names():
    return list(enumerate(names))











