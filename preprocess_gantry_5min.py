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
parser.add_argument("--config", default="/data/Huawei/flow_prediction/ASTGNN_Lv/configurations/gantry_avgspeed_20230801-20240124.conf", type=str, help="configuration file path")
args = parser.parse_args()

# read config
config = configparser.ConfigParser()
config_path = os.path.join('configurations', args.config)
print('Read configuration file: %s' % config_path, flush=True)
config.read(config_path)

# parse config: features and date
valid_feat = ['avgspeed']

feat_str = config['Data']['features']
feat = feat_str.split('_')

if not all([f in valid_feat for f in feat]):
    raise ValueError('Invalid feature type, should be one of %s' % valid_feat)

# 起始时间和结束时间
start_date = config['Data']['start_date'] + " 00:00:00"
end_date = config['Data']['end_date'] + " 23:59:00"

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

# NUM_GANTRY = 208
gantry_df = pd.read_csv('/data/Huawei/flow_prediction/data_process/dwd_ldj_fact_gantrytraveldate_5min平均速度.csv')
gantry_id_df = pd.read_csv('/data/Huawei/flow_prediction/ASTGNN_20240516/raw_data/去掉高缺失率数据后的208个门架.csv')
# gantry_info = pd.read_csv(ws + '/raw_data/去掉高缺失率数据后的208个门架.csv')
# adj_matrix = pd.read_csv(ws + '/raw_data/去掉高缺失率数据后的375个门架收费站邻接矩阵.csv')

if not os.path.exists("%s/data/gantry/%s/%s" % (ws, feat_str, date_str)):
    os.makedirs("%s/data/gantry/%s/%s" % (ws, feat_str, date_str))

# gantry_df = gantry_df[(gantry_df['ownername'] == '昆明西管理处') | (gantry_df['ownername'] == '昆明东管理处')]
gantry_df.fillna(0)

gantry_df['gantryid'] = gantry_df['gantrystart']
gantry_df['gantryid'] = gantry_df['gantryid'].str.cat(gantry_df['gantryend'], sep='_')

gantry_df['gantryname'] = gantry_df['gantrystartname']
gantry_df['gantryname'] = gantry_df['gantryname'].str.cat(gantry_df['gantryendname'], sep='_')

gantry_id_list = []
id_list = gantry_id_df['gantryid'].unique()
for i in id_list:
    for j in id_list:
        gantry_id_list.append(i + '_' + j)

gantry_df = gantry_df[gantry_df['gantryid'].isin(gantry_id_list)]


# gantry_gantryid_list = list(gantry_name_id_dic.values())
# gantry_id_name_dic = dict(zip(gantry_name_id_dic.values(), range(NUM_GANTRY)))
# gantry_gantryid_array = np.array(gantry_gantryid_list)
# gantry_gantryid_array = gantry_gantryid_array.reshape(-1, 1)
# gantry_id_df = pd.DataFrame(gantry_gantryid_array)
# gantry_id_df.to_csv(ws + '/data/gantry/%s/%s/gantryid_df.csv' % (feat_str, date_str),header=0, index=0)

gantry_flow = deepcopy(gantry_df[['gantryid','gantryname','statisticaltime', 'avgspeed', 'flow']])
gantry_flow['statisticaltime'] = pd.to_datetime(gantry_flow['statisticaltime'])
gantry_flow = gantry_flow[(gantry_flow['statisticaltime'] >= start_date) & (gantry_flow['statisticaltime'] <= end_date)]
gantry_flow['strtime'] = gantry_flow['statisticaltime'].dt.strftime("%H%M")

# gantry_id_list = gantry_df['gantryid'].unique()
gantry_id_list = gantry_flow['gantryid'].unique()

print("num_node: %d" % len(gantry_id_list))
print("max date: {}, min date: {}".format(gantry_flow['statisticaltime'].max(), gantry_flow['statisticaltime'].min()))


gantry_flow_wo_gantryname = deepcopy(gantry_flow[['gantryid', 'strtime', 'statisticaltime','avgspeed', 'flow']])

# gantry_index
idx_gantryid = idx(gantry_flow_wo_gantryname, 'gantryid')
idx_gantry_strtime = idx(gantry_flow_wo_gantryname, 'strtime')
idx_gantry_sttime = idx(gantry_flow_wo_gantryname, 'statisticaltime')
idx_gantry_speed = idx(gantry_flow_wo_gantryname, 'avgspeed')
idx_gantry_flow = idx(gantry_flow_wo_gantryname, 'flow')

gantry_flow_np = gantry_flow_wo_gantryname.values

# 定义步长为5min
step = timedelta(minutes=5)

# 生成日期时间列表
date_list = []
current_date = start_date
while current_date <= end_date:
    date_list.append(current_date)
    current_date += step

#计算每个节点在每个时间点的平均流量
avg_flow = {}


# for i in tqdm(gantry_id_list):
#     cur_dic = []
#     for h in range(24):
#         for m in range(0, 60, 5):
#             j = "%02d%02d" % (h, m)
#             gantry_id_info = gantry_flow_np[gantry_flow_np[:, idx_gantryid]==i]
#             gantry_id_time_info = gantry_id_info[gantry_id_info[:, idx_gantry_strtime] == j]
#             speed = gantry_id_time_info[:, idx_gantry_speed].mean()
#             # flow = gantry_id_time_info[:, idx_gantry_flow].mean()
#             # cur_dic.append((speed if not np.isnan(speed) else 0, flow if not np.isnan(flow) else 0))
#             cur_dic.append((speed if not np.isnan(speed) else 0, 0))  # 不计算flow
#     avg_flow[i] = cur_dic


# import pickle
# # with open(ws + '/data/gantry/%s/%s/avg_flow_gantry.pkl' % (feat_str, date_str), 'wb') as f:
# #     pickle.dump(avg_flow, f)

# with open(ws + '/data/gantry/%s/%s/avg_flow_gantry.pkl' % (feat_str, date_str), 'rb') as f:
#     avg_flow = pickle.load(f)

holidays = pd.to_datetime([
    '2023-01-01', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27', # 春节
    '2023-04-05', # 清明节
    '2023-05-01', '2023-05-02', '2023-05-03', # 劳动节
    '2023-06-22', '2023-06-23', '2023-06-24', # 端午节
    '2023-09-29', '2023-09-30', '2023-10-01', '2023-10-02', '2023-10-03', '2023-10-04', '2023-10-05', '2023-10-06', # 国庆节 & 中秋节
    '2024-01-01'
]).date

holiday_info = np.zeros((len(date_list), 1))

# holidays
# for i in tqdm(range(len(date_list))):
#     if date_list[i].date() in holidays:
#         holiday_info[i] = 1

# flow_data_3 = -np.ones((len(date_list), len(gantry_id_list), len(feat) + 1))
flow_data_3 = -np.ones((len(date_list), len(gantry_id_list), len(feat)))
# id_name_dic_new = dict(zip(gantry_id_name_dic.values(), gantry_id_name_dic.keys()))

for i in tqdm(gantry_flow_wo_gantryname.itertuples()):
    if i[idx_gantry_sttime + 1] < start_date or i[idx_gantry_sttime + 1] > end_date:
        continue

    if i[idx_gantryid + 1] in gantry_id_list:
        date = int((i[idx_gantry_sttime + 1] - start_date).total_seconds()//300)
        station = np.where(gantry_id_list == i[idx_gantryid + 1])[0][0]

        idx_ = 0
        if "avgspeed" in feat:
            flow_data_3[date][station][idx_] = i[idx_gantry_speed + 1]
            idx_ += 1
        if "flow" in feat:
            flow_data_3[date][station][idx_] = i[idx_gantry_flow + 1]
            idx_ += 1

        # flow_data_3[date][station][idx_] = datetime.timestamp(i[idx_gantry_sttime + 1])  # 时间戳


# 给训练集补值
# i = 0
# while i < len(date_list) * 0.6 + 12 + 7 * 24 * 12:
#     for j in range(len(gantry_id_list)):
#         if flow_data_3[i][j][0] < 0.:
#             step = int(i % 288)
#             # station = id_name_dic_new[j]

#             idx_= 0
#             if "avgspeed" in feat:
#                 flow_data_3[i][j][idx_] = avg_flow[gantry_id_list[j]][step][0]
#                 idx_ += 1
#             if "flow" in feat:
#                 flow_data_3[i][j][idx_] = avg_flow[gantry_id_list[j]][step][1]
#                 idx_ += 1
#             # flow_data_3[i][j][idx_] = float(step)  # 时间戳

#     i = i + 1

# adj_matrix = adj_matrix.drop('Unnamed: 0',axis=1)

# num_id_dic = {}
# for i,j in enumerate(gantry_name_id_dic.keys()):
#     num_id_dic[i] = gantry_name_id_dic[j]

# adj_matrix.rename(columns=gantry_name_id_dic, index=num_id_dic, inplace=True)

adj_csv = []
for i in range(len(gantry_id_list)):
    for j in range(len(gantry_id_list)):
        # if adj_matrix.loc[i,j] == 1:
        adj_csv.append([i,j,1])

adj_csv = pd.DataFrame(adj_csv, columns=['from','to','distance'])
adj_csv.to_csv(ws + '/data/gantry/%s/%s/gantry_adj.csv' % (feat_str, date_str),index=False)
np.save(ws + '/data/gantry/%s/%s/flow_data_gantry.npy' % (feat_str, date_str),flow_data_3)
np.save(ws + '/data/gantry/%s/%s/holidayinfo.npy' % (feat_str, date_str), holiday_info)
print('saved ~~')