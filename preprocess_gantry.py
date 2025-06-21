import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from copy import deepcopy
import os
from tqdm import tqdm
import configparser
import argparse

# config path
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="/data/Huawei/flow_prediction/ASTGNN_20240519/configurations/gantry_avgspeed_20240126-20240425.conf", type=str, help="configuration file path")
args = parser.parse_args()

# read config
config = configparser.ConfigParser()
config_path = os.path.join('configurations', args.config)
print('Read configuration file: %s' % config_path, flush=True)
config.read(config_path)

# parse config: features and date
valid_feat = ['vehicle', 'coach', 'truck', 'coach1', 'coach2', 'coach3', 'coach4', 'truck1', 'truck2', 'truck3', 'truck4', 'truck5', 'truck6']
# valid_feat = ['vehicle', 'coach', 'truck']
# valid_feat = ['vehicle', 'coach', 'truck']

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

NUM_GANTRY = 208
gantry_info = pd.read_csv(ws + '/raw_data/去掉高缺失率数据后的208个门架.csv', encoding="utf-8")
gantry_df = pd.read_hdf(ws + '/raw_data/gantry_flow.hdf', key='df')
adj_matrix = pd.read_csv(ws + '/raw_data/去掉高缺失率数据后的375个门架收费站邻接矩阵.csv')

if not os.path.exists("%s/data/gantry/%s/%s" % (ws, feat_str, date_str)):
    os.makedirs("%s/data/gantry/%s/%s" % (ws, feat_str, date_str))

gantry_name_id_dic = {}
for i in gantry_info['ORGNAME']:
    gantry_name_id_dic[i] = gantry_info[gantry_info['ORGNAME']==i]['gantryid'].iloc[0]

gantry_gantryid_list = list(gantry_name_id_dic.values())
gantry_id_name_dic = dict(zip(gantry_name_id_dic.values(), range(NUM_GANTRY)))
gantry_gantryid_array = np.array(gantry_gantryid_list)
gantry_gantryid_array = gantry_gantryid_array.reshape(-1, 1)
gantry_id_df = pd.DataFrame(gantry_gantryid_array)
gantry_id_df.to_csv(ws + '/data/gantry/%s/%s/gantryid_df.csv' % (feat_str, date_str),header=0, index=0)

gantry_flow = gantry_df[['gantryid','gantryname','statisticalhour', 'vehicleflow', 'coachvehicleflow','truckvehicleflow',
                         'coachvehtypeflow1', 'coachvehtypeflow2', 'coachvehtypeflow3', 'coachvehtypeflow4',
                         'truckvehicleflow1', 'truckvehicleflow2', 'truckvehicleflow3', 'truckvehicleflow4',
                         'truckvehicleflow5', 'truckvehicleflow6']]

gantry_flow['statisticalhour'] = pd.to_datetime(gantry_flow['statisticalhour'])
gantry_flow['hour'] = gantry_flow['statisticalhour'].dt.hour
gantry_flow_wo_stathour = deepcopy(gantry_flow[['gantryid', 'statisticalhour', 'hour','vehicleflow', 'coachvehicleflow','truckvehicleflow',
                                                'coachvehtypeflow1', 'coachvehtypeflow2', 'coachvehtypeflow3',
                                                'coachvehtypeflow4',
                                                'truckvehicleflow1', 'truckvehicleflow2', 'truckvehicleflow3',
                                                'truckvehicleflow4',
                                                'truckvehicleflow5', 'truckvehicleflow6'
                                                ]])
# gantry_index
idx_gantryid = idx(gantry_flow_wo_stathour, 'gantryid')
idx_gantry_statisticalhour = idx(gantry_flow_wo_stathour, 'statisticalhour')
idx_gantry_hour = idx(gantry_flow_wo_stathour, 'hour')
idx_gantry_sumflow = idx(gantry_flow_wo_stathour, 'vehicleflow')
idx_gantry_coachflow = idx(gantry_flow_wo_stathour, 'coachvehicleflow')
idx_gantry_truckflow = idx(gantry_flow_wo_stathour, 'truckvehicleflow')

idx_gantry_coach1 = idx(gantry_flow_wo_stathour, 'coachvehtypeflow1')
idx_gantry_coach2 = idx(gantry_flow_wo_stathour, 'coachvehtypeflow2')
idx_gantry_coach3 = idx(gantry_flow_wo_stathour, 'coachvehtypeflow3')
idx_gantry_coach4 = idx(gantry_flow_wo_stathour, 'coachvehtypeflow4')

idx_gantry_truck1 = idx(gantry_flow_wo_stathour, 'truckvehicleflow1')
idx_gantry_truck2 = idx(gantry_flow_wo_stathour, 'truckvehicleflow2')
idx_gantry_truck3 = idx(gantry_flow_wo_stathour, 'truckvehicleflow3')
idx_gantry_truck4 = idx(gantry_flow_wo_stathour, 'truckvehicleflow4')
idx_gantry_truck5 = idx(gantry_flow_wo_stathour, 'truckvehicleflow5')
idx_gantry_truck6 = idx(gantry_flow_wo_stathour, 'truckvehicleflow6')

gantry_flow_np = gantry_flow_wo_stathour.values

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

for i in tqdm(gantry_gantryid_list):
    cur_dic = []
    for j in range(24):
        gantry_id_info = gantry_flow_np[gantry_flow_np[:, idx_gantryid]==i]
        gantry_id_hour_info = gantry_id_info[gantry_id_info[:, idx_gantry_hour] == j]
        sum_flow = gantry_id_hour_info[:, idx_gantry_sumflow].mean()
        en_coach_flow = gantry_id_hour_info[:, idx_gantry_coachflow].mean()
        en_truck_flow = gantry_id_hour_info[:, idx_gantry_truckflow].mean()
        coach1 = gantry_id_hour_info[:, idx_gantry_coach1].mean()
        coach2 = gantry_id_hour_info[:, idx_gantry_coach2].mean()
        coach3 = gantry_id_hour_info[:, idx_gantry_coach3].mean()
        coach4 = gantry_id_hour_info[:, idx_gantry_coach4].mean()

        truck1 = gantry_id_hour_info[:, idx_gantry_truck1].mean()
        truck2 = gantry_id_hour_info[:, idx_gantry_truck2].mean()
        truck3 = gantry_id_hour_info[:, idx_gantry_truck3].mean()
        truck4 = gantry_id_hour_info[:, idx_gantry_truck4].mean()
        truck5 = gantry_id_hour_info[:, idx_gantry_truck5].mean()
        truck6 = gantry_id_hour_info[:, idx_gantry_truck6].mean()

        cur_dic.append((sum_flow, en_coach_flow, en_truck_flow, coach1, coach2, coach3, coach4, truck1, truck2, truck3, truck4, truck5, truck6))
    avg_flow[i] = cur_dic
#
#
import pickle
with open(ws + '/data/gantry/%s/%s/avg_flow_gantry.pkl' % (feat_str, date_str), 'wb') as f:
    pickle.dump(avg_flow, f)

with open(ws + '/data/gantry/%s/%s/avg_flow_gantry.pkl' % (feat_str, date_str), 'rb') as f:
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

flow_gantry = -np.ones((len(date_list), len(gantry_gantryid_list), len(feat) + 1))
id_name_dic_new = dict(zip(gantry_id_name_dic.values(), gantry_id_name_dic.keys()))

for i in tqdm(gantry_flow_wo_stathour.itertuples()):
    if i[idx_gantry_statisticalhour + 1] < start_date or i[idx_gantry_statisticalhour + 1] > end_date:
        continue

    if i[idx_gantryid + 1] in gantry_id_name_dic.keys():
        date = int((i[idx_gantry_statisticalhour + 1] - start_date).total_seconds()//3600)
        station = gantry_id_name_dic[i[idx_gantryid + 1]]

        idx_ = 0
        if "vehicle" in feat:
            flow_gantry[date][station][idx_] = i[idx_gantry_sumflow + 1]
            idx_ += 1
        if "coach" in feat:
            flow_gantry[date][station][idx_] = i[idx_gantry_coachflow + 1]
            idx_ += 1
        if "truck" in feat:
            flow_gantry[date][station][idx_] = i[idx_gantry_truckflow + 1]
            idx_ += 1
        if "coach1" in feat:
            flow_gantry[date][station][idx_] = i[idx_gantry_coach1 + 1]
            idx_ += 1
        if "coach2" in feat:
            flow_gantry[date][station][idx_] = i[idx_gantry_coach2 + 1]
            idx_ += 1
        if "coach3" in feat:
            flow_gantry[date][station][idx_] = i[idx_gantry_coach3 + 1]
            idx_ += 1
        if "coach4" in feat:
            flow_gantry[date][station][idx_] = i[idx_gantry_coach4 + 1]
            idx_ += 1
        if "truck1" in feat:
            flow_gantry[date][station][idx_] = i[idx_gantry_truck1 + 1]
            idx_ += 1
        if "truck2" in feat:
            flow_gantry[date][station][idx_] = i[idx_gantry_truck2 + 1]
            idx_ += 1
        if "truck3" in feat:
            flow_gantry[date][station][idx_] = i[idx_gantry_truck3 + 1]
            idx_ += 1
        if "truck4" in feat:
            flow_gantry[date][station][idx_] = i[idx_gantry_truck4 + 1]
            idx_ += 1
        if "truck5" in feat:
            flow_gantry[date][station][idx_] = i[idx_gantry_truck5 + 1]
            idx_ += 1
        if "truck6" in feat:
            flow_gantry[date][station][idx_] = i[idx_gantry_truck6 + 1]
            idx_ += 1

        flow_gantry[date][station][idx_] = i[idx_gantry_hour + 1]


i = 0
while i < len(flow_gantry)*0.6+180:
    for j in range(len(flow_gantry[i])):
        if flow_gantry[i][j][0] < 0.:
            hour = int(i % 24)
            station = id_name_dic_new[j]
            idx_= 0
            if "vehicle" in feat:
                flow_gantry[i][j][idx_] = avg_flow[station][hour][0]
                idx_ += 1
            if "coach" in feat:
                flow_gantry[i][j][idx_] = avg_flow[station][hour][1]
                idx_ += 1
            if "truck" in feat:
                flow_gantry[i][j][idx_] = avg_flow[station][hour][2]
                idx_ += 1
            if "coach1" in feat:
                flow_gantry[i][j][idx_] =  avg_flow[station][hour][3]
                idx_ += 1
            if "coach2" in feat:
                flow_gantry[i][j][idx_] =  avg_flow[station][hour][4]
                idx_ += 1
            if "coach3" in feat:
                flow_gantry[i][j][idx_] =  avg_flow[station][hour][5]
                idx_ += 1
            if "coach4" in feat:
                flow_gantry[i][j][idx_] =  avg_flow[station][hour][6]
                idx_ += 1
            if "truck1" in feat:
                flow_gantry[i][j][idx_] =  avg_flow[station][hour][7]
                idx_ += 1
            if "truck2" in feat:
                flow_gantry[i][j][idx_] =  avg_flow[station][hour][8]
                idx_ += 1
            if "truck3" in feat:
                flow_gantry[i][j][idx_] =  avg_flow[station][hour][9]
                idx_ += 1
            if "truck4" in feat:
                flow_gantry[i][j][idx_] =  avg_flow[station][hour][10]
                idx_ += 1
            if "truck5" in feat:
                flow_gantry[i][j][idx_] =  avg_flow[station][hour][11]
                idx_ += 1
            if "truck6" in feat:
                flow_gantry[i][j][idx_] =  avg_flow[station][hour][12]
                idx_ += 1
            flow_gantry[i][j][idx_] = float(hour)

    i = i + 1

adj_matrix = adj_matrix.drop('Unnamed: 0',axis=1)

num_id_dic = {}
for i,j in enumerate(gantry_name_id_dic.keys()):
    num_id_dic[i] = gantry_name_id_dic[j]

adj_matrix.rename(columns=gantry_name_id_dic, index=num_id_dic, inplace=True)

adj_csv = []
for i in gantry_gantryid_list:
    for j in gantry_gantryid_list:
        if adj_matrix.loc[i,j] == 1:
            adj_csv.append([i,j,1])

adj_csv = pd.DataFrame(adj_csv, columns=['from','to','distance'])
adj_csv.to_csv(ws + '/data/gantry/%s/%s/gantry_adj.csv' % (feat_str, date_str),index=False)
np.save(ws + '/data/gantry/%s/%s/flow_data_gantry.npy' % (feat_str, date_str),flow_gantry)
np.save(ws + '/data/gantry/%s/%s/holidayinfo.npy' % (feat_str, date_str), holiday_info)
np.save(ws + '/data/gantry/%s/%s/weekendifno.npy' % (feat_str, date_str), weekend_info)
print('saved ~~')