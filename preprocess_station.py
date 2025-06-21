import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import os
import argparse
import configparser

# config path
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="station_enflow_exflow_sumflow_sumcoach_sumtruck_coach1_coach2_coach3_coach4_truck1_truck2_truck3_truck4_truck5_truck6_20230101_20240124.conf", type=str, help="configuration file path")
args = parser.parse_args()

# read config
config = configparser.ConfigParser()
config_path = os.path.join('configurations', args.config)
print('Read configuration file: %s' % config_path, flush=True)
config.read(config_path)

# parse config: features and date
valid_feat = ['enflow', 'exflow', 'sumflow', 'sumcoach', 'sumtruck', 'coach1', 'coach2', 'coach3', 'coach4', 'truck1', 'truck2', 'truck3', 'truck4', 'truck5', 'truck6']

feat_str = config['Data']['features']
feat = feat_str.split('_')

if not all([f in valid_feat for f in feat]):
    raise ValueError('Invalid feature type, should be one of %s' % valid_feat)

# 起始时间和结束时间
start_date = config['Data']['start_date'] + " 00:00:00"
end_date = config['Data']['end_date'] + " 23:00:00"

start_date = datetime.strptime(start_date, "%Y%m%d %H:%M:%S")
end_date = datetime.strptime(end_date, "%Y%m%d %H:%M:%S")

date_str = config['Data']['start_date'] + "_" + config['Data']['end_date']

# ====== Prepare finish ======= #

def idx(df, col_name):
    _idx_ = list(df.columns).index(col_name)
    return _idx_

def get_workspace():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    return file

ws =  get_workspace()
STATION_NUM = 125
station_df = pd.read_hdf(ws + '/raw_data/station_flow.hdf', key='df')
station_info = pd.read_csv(ws + '/raw_data/去掉高缺失率数据后的125个收费站.csv')
adj_matrix = pd.read_csv(ws + '/raw_data/adjacency_matrix_station_去掉撮科_0328.csv')

if not os.path.exists("%s/data/station/%s/%s" % (ws, feat_str, date_str)):
    os.makedirs("%s/data/station/%s/%s" % (ws, feat_str, date_str))

station_name_id_dic = {}
for i in adj_matrix['Unnamed: 0'].iloc[:STATION_NUM]:
    station_name_id_dic[i] = int(station_info[station_info['STATIONNAME']==i]['STATIONID'].iloc[0])

station_stationid_list = list(station_name_id_dic.values())
station_id_name_dic = dict(zip(station_name_id_dic.values(), range(STATION_NUM)))
station_stationid_array = np.array(station_stationid_list)
station_stationid_array = station_stationid_array.reshape(-1, 1)
station_id_df = pd.DataFrame(station_stationid_array)
station_id_df.to_csv(ws + '/data/station/%s/%s/stationid_df.csv' % (feat_str, date_str), header=0, index=0)


station_flow = station_df[['stationid','statisticalhour','enflow','exflow', 'encoachflow1', 'encoachflow2', 'encoachflow3', 'encoachflow4', 'entruckflow1', 'entruckflow2', 'entruckflow3', 'entruckflow4', 'entruckflow5', 'entruckflow6', 'excoachflow1','excoachflow2','excoachflow3','excoachflow4','extruckflow1','extruckflow2','extruckflow3','extruckflow4','extruckflow5','extruckflow6']]
station_flow['statisticalhour'] = pd.to_datetime(station_flow['statisticalhour'])
station_flow['hour'] = station_flow['statisticalhour'].dt.hour
station_flow['sumcoach'] = station_flow['encoachflow1'] + station_flow['encoachflow2'] + station_flow['encoachflow3'] + station_flow['encoachflow4'] + station_flow['excoachflow1'] + station_flow['excoachflow2'] + station_flow['excoachflow3'] + station_flow['excoachflow4']
station_flow['sumtruck'] = station_flow['entruckflow1'] + station_flow['entruckflow2'] + station_flow['entruckflow3'] + station_flow['entruckflow4'] + station_flow['entruckflow5'] + station_flow['entruckflow6'] + station_flow['extruckflow1'] + station_flow['extruckflow2'] + station_flow['extruckflow3'] + station_flow['extruckflow4'] + station_flow['extruckflow5'] + station_flow['extruckflow6']
station_flow['coach1'] = station_flow['encoachflow1'] + station_flow['excoachflow1']
station_flow['coach2'] = station_flow['encoachflow2'] + station_flow['excoachflow2']
station_flow['coach3'] = station_flow['encoachflow3'] + station_flow['excoachflow3']
station_flow['coach4'] = station_flow['encoachflow4'] + station_flow['excoachflow4']

station_flow['truck1'] = station_flow['entruckflow1'] + station_flow['extruckflow1']
station_flow['truck2'] = station_flow['entruckflow2'] + station_flow['extruckflow2']
station_flow['truck3'] = station_flow['entruckflow3'] + station_flow['extruckflow3']
station_flow['truck4'] = station_flow['entruckflow4'] + station_flow['extruckflow4']
station_flow['truck5'] = station_flow['entruckflow5'] + station_flow['extruckflow5']
station_flow['truck6'] = station_flow['entruckflow6'] + station_flow['extruckflow6']

station_flow_wo_stathour = deepcopy(station_flow[['stationid', 'statisticalhour', 'hour','enflow','exflow', 'sumcoach', 'sumtruck',
                                                  'coach1', 'coach2', 'coach3', 'coach4', 'truck1', 'truck2', 'truck3', 'truck4', 'truck5', 'truck6',
                                                  'encoachflow1', 'encoachflow2', 'encoachflow3', 'encoachflow4',
                                                  'entruckflow1', 'entruckflow2', 'entruckflow3', 'entruckflow4', 'entruckflow5', 'entruckflow6',
                                                  'excoachflow1','excoachflow2','excoachflow3','excoachflow4',
                                                  'extruckflow1','extruckflow2','extruckflow3','extruckflow4','extruckflow5','extruckflow6']])

#index
idx_stationid = idx(station_flow_wo_stathour, 'stationid')
idx_statisticalhour = idx(station_flow_wo_stathour, 'statisticalhour')
idx_hour = idx(station_flow_wo_stathour, 'hour')
idx_enflow = idx(station_flow_wo_stathour, 'enflow')
idx_exflow = idx(station_flow_wo_stathour, 'exflow')
idx_sumcoach = idx(station_flow_wo_stathour, 'sumcoach')
idx_sumtruck = idx(station_flow_wo_stathour, 'sumtruck')

idx_coach1 = idx(station_flow_wo_stathour, 'coach1')
idx_coach2 = idx(station_flow_wo_stathour, 'coach2')
idx_coach3 = idx(station_flow_wo_stathour, 'coach3')
idx_coach4 = idx(station_flow_wo_stathour, 'coach4')

idx_truck1 = idx(station_flow_wo_stathour, 'truck1')
idx_truck2 = idx(station_flow_wo_stathour, 'truck2')
idx_truck3 = idx(station_flow_wo_stathour, 'truck3')
idx_truck4 = idx(station_flow_wo_stathour, 'truck4')
idx_truck5 = idx(station_flow_wo_stathour, 'truck5')
idx_truck6 = idx(station_flow_wo_stathour, 'truck6')

idx_encoachflow1 = idx(station_flow_wo_stathour, 'encoachflow1')
idx_encoachflow2 = idx(station_flow_wo_stathour, 'encoachflow2')
idx_encoachflow3 = idx(station_flow_wo_stathour, 'encoachflow3')
idx_encoachflow4 = idx(station_flow_wo_stathour, 'encoachflow4')

idx_entruckflow1 = idx(station_flow_wo_stathour, 'entruckflow1')
idx_entruckflow2 = idx(station_flow_wo_stathour, 'entruckflow2')
idx_entruckflow3 = idx(station_flow_wo_stathour, 'entruckflow3')
idx_entruckflow4 = idx(station_flow_wo_stathour, 'entruckflow4')
idx_entruckflow5 = idx(station_flow_wo_stathour, 'entruckflow5')
idx_entruckflow6 = idx(station_flow_wo_stathour, 'entruckflow6')

idx_excoachflow1 = idx(station_flow_wo_stathour, 'excoachflow1')
idx_excoachflow2 = idx(station_flow_wo_stathour, 'excoachflow2')
idx_excoachflow3 = idx(station_flow_wo_stathour, 'excoachflow3')
idx_excoachflow4 = idx(station_flow_wo_stathour, 'excoachflow4')

idx_extruckflow1 = idx(station_flow_wo_stathour, 'extruckflow1')
idx_extruckflow2 = idx(station_flow_wo_stathour, 'extruckflow2')
idx_extruckflow3 = idx(station_flow_wo_stathour, 'extruckflow3')
idx_extruckflow4 = idx(station_flow_wo_stathour, 'extruckflow4')
idx_extruckflow5 = idx(station_flow_wo_stathour, 'extruckflow5')
idx_extruckflow6 = idx(station_flow_wo_stathour, 'extruckflow6')

station_flow_np = station_flow_wo_stathour.values

# 定义步长为1小时
step = timedelta(hours=1)

# 生成日期时间列表
date_list = []
current_date = start_date
while current_date <= end_date:
    date_list.append(current_date)
    current_date += step

#计算每个节点在每个时间点的平均流量
avg_flow = {}
for i in tqdm(station_stationid_list):
    cur_dic = []
    for j in range(24):
        station_id_info = station_flow_np[station_flow_np[:, idx_stationid]==i]
        station_id_hour_info = station_id_info[station_id_info[:, idx_hour] == j]
        en_flow = station_id_hour_info[:, idx_enflow].mean()
        ex_flow = station_id_hour_info[:, idx_exflow].mean()
        coachflow = station_id_hour_info[:, idx_sumcoach].mean()
        truckflow = station_id_hour_info[:, idx_sumtruck].mean()
        coach1 = station_id_hour_info[:, idx_coach1].mean()
        coach2 = station_id_hour_info[:, idx_coach2].mean()
        coach3 = station_id_hour_info[:, idx_coach3].mean()
        coach4 = station_id_hour_info[:, idx_coach4].mean()

        truck1 = station_id_hour_info[:, idx_truck1].mean()
        truck2 = station_id_hour_info[:, idx_truck2].mean()
        truck3 = station_id_hour_info[:, idx_truck3].mean()
        truck4 = station_id_hour_info[:, idx_truck4].mean()
        truck5 = station_id_hour_info[:, idx_truck5].mean()
        truck6 = station_id_hour_info[:, idx_truck6].mean()

        cur_dic.append((en_flow, ex_flow, en_flow + ex_flow, coachflow, truckflow, coach1, coach2, coach3, coach4, truck1, truck2, truck3, truck4, truck5, truck6))
    avg_flow[i] = cur_dic


import pickle
with open(ws + '/data/station/%s/%s/avg_flow_station.pkl' % (feat_str, date_str), 'wb') as f:
    pickle.dump(avg_flow, f)

with open(ws + '/data/station/%s/%s/avg_flow_station.pkl' % (feat_str, date_str), 'rb') as f:
    avg_flow = pickle.load(f)

holidays = pd.to_datetime([
    '2023-01-01', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27', # 春节
    '2023-04-05', # 清明节
    '2023-05-01', '2023-05-02', '2023-05-03', # 劳动节
    '2023-06-22', '2023-06-23', '2023-06-24', # 端午节
    '2023-09-29', '2023-09-30', '2023-10-01', '2023-10-02', '2023-10-03', '2023-10-04', '2023-10-05', '2023-10-06', # 国庆节 & 中秋节
    '2024-01-01'
]).date

holiday_info = np.zeros((len(date_list), 1))

for i in tqdm(range(len(date_list))):
    if date_list[i].date() in holidays:
        holiday_info[i] = 1

weekend_info = np.zeros((len(date_list), 1))

for i in tqdm(range(len(date_list))):
    if date_list[i].weekday() in [5, 6]:  # 5: Saturday, 6: Sunday
        weekend_info[i] = 1

flow_data = -np.ones((len(date_list), len(station_stationid_list),len(feat) + 1)) # [b, N, F + 1], 各种流量，小时时间戳
id_name_dic_new = dict(zip(station_id_name_dic.values(), station_id_name_dic.keys()))
for i in tqdm(station_flow_wo_stathour.itertuples()):
    if i[idx_statisticalhour + 1] < start_date or i[idx_statisticalhour + 1] > end_date:
        continue

    if i[idx_stationid + 1] in station_id_name_dic.keys():
        date = int((i[idx_statisticalhour + 1] - start_date).total_seconds()//3600)
        station = station_id_name_dic[i[idx_stationid + 1]]
        idx_ = 0
        if 'enflow' in feat:
            flow_data[date][station][idx_] = i[idx_enflow + 1]
            idx_ += 1
        if 'exflow' in feat:
            flow_data[date][station][idx_] = i[idx_exflow + 1]
            idx_ += 1
        if 'sumflow' in feat:
            flow_data[date][station][idx_] = i[idx_enflow + 1] + i[idx_exflow + 1] # enflow + exflow
            idx_ += 1
        if 'sumcoach' in feat:
            flow_data[date][station][idx_] = i[idx_sumcoach + 1]
            idx_ += 1
        if 'sumtruck' in feat:
            flow_data[date][station][idx_] = i[idx_sumtruck + 1]
            idx_ += 1
        if 'coach1' in feat:
            flow_data[date][station][idx_] = i[idx_coach1 + 1]
            idx_ += 1
        if 'coach2' in feat:
            flow_data[date][station][idx_] = i[idx_coach2 + 1]
            idx_ += 1
        if 'coach3' in feat:
            flow_data[date][station][idx_] = i[idx_coach3 + 1]
            idx_ += 1
        if 'coach4' in feat:
            flow_data[date][station][idx_] = i[idx_coach4 + 1]
            idx_ += 1
        if 'truck1' in feat:
            flow_data[date][station][idx_] = i[idx_truck1 + 1]
            idx_ += 1
        if 'truck2' in feat:
            flow_data[date][station][idx_] = i[idx_truck2 + 1]
            idx_ += 1
        if 'truck3' in feat:
            flow_data[date][station][idx_] = i[idx_truck3 + 1]
            idx_ += 1
        if 'truck4' in feat:
            flow_data[date][station][idx_] = i[idx_truck4 + 1]
            idx_ += 1
        if 'truck5' in feat:
            flow_data[date][station][idx_] = i[idx_truck5 + 1]
            idx_ += 1
        if 'truck6' in feat:
            flow_data[date][station][idx_] = i[idx_truck6 + 1]
            idx_ += 1

        flow_data[date][station][idx_] = i[idx_statisticalhour + 1].hour


#训练集补充缺失值
i = 0
while i < len(flow_data) * 0.6 + 12 + 7 * 24:
    for j in range(len(flow_data[i])):
        if flow_data[i][j][0] < 0.:
            hour = int(i % 24)
            station = id_name_dic_new[j]
            idx_ = 0 # feature
            if 'enflow' in feat:
                flow_data[i][j][idx_] = avg_flow[station][hour][0]
                idx_ += 1
            if 'exflow' in feat:
                flow_data[i][j][idx_] = avg_flow[station][hour][1]
                idx_ += 1
            if 'sumflow' in feat:
                flow_data[i][j][idx_] = avg_flow[station][hour][2]
                idx_ += 1
            if 'sumcoach' in feat:
                flow_data[i][j][idx_] = avg_flow[station][hour][3]
                idx_ += 1
            if 'sumtruck' in feat:
                flow_data[i][j][idx_] = avg_flow[station][hour][4]
                idx_ += 1
            if 'coach1' in feat:
                flow_data[i][j][idx_] = avg_flow[station][hour][5]
                idx_ += 1
            if 'coach2' in feat:
                flow_data[i][j][idx_] = avg_flow[station][hour][6]
                idx_ += 1
            if 'coach3' in feat:
                flow_data[i][j][idx_] = avg_flow[station][hour][7]
                idx_ += 1
            if 'coach4' in feat:
                flow_data[i][j][idx_] = avg_flow[station][hour][8]
                idx_ += 1
            if 'truck1' in feat:
                flow_data[i][j][idx_] = avg_flow[station][hour][9]
                idx_ += 1
            if 'truck2' in feat:
                flow_data[i][j][idx_] = avg_flow[station][hour][10]
                idx_ += 1
            if 'truck3' in feat:
                flow_data[i][j][idx_] = avg_flow[station][hour][11]
                idx_ += 1
            if 'truck4' in feat:
                flow_data[i][j][idx_] = avg_flow[station][hour][12]
                idx_ += 1
            if 'truck5' in feat:
                flow_data[i][j][idx_] = avg_flow[station][hour][13]
                idx_ += 1
            if 'truck6' in feat:
                flow_data[i][j][idx_] = avg_flow[station][hour][14]
                idx_ += 1
            flow_data[i][j][idx_] = float(hour)

    i = i + 1

adj_matrix = adj_matrix.drop('Unnamed: 0',axis=1)

num_id_dic = {}
for i,j in enumerate(station_id_name_dic.keys()):
    num_id_dic[i] = j

adj_matrix.rename(columns=station_name_id_dic, index=num_id_dic, inplace=True)

adj_csv = []
for i in station_stationid_list:
    for j in station_stationid_list:
        if adj_matrix.loc[i,j] == 1:
            adj_csv.append([i,j,1])


adj_csv = pd.DataFrame(adj_csv, columns=['from','to','distance'])
adj_csv.to_csv(ws + '/data/station/%s/%s/station_adj.csv' % (feat_str, date_str),index=False)
np.save(ws + '/data/station/%s/%s/flow_data_station.npy' % (feat_str, date_str), flow_data)
np.save(ws + '/data/station/%s/%s/holidayinfo_station.npy' % (feat_str, date_str),holiday_info)
np.save(ws + '/data/gantry/%s/%s/weekendifno.npy' % (feat_str, date_str), weekend_info)
print('saved ~~')