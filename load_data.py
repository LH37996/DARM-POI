import numpy as np
from collections import defaultdict
import json
import pandas as pd
from ast import literal_eval


class Data:
    def __init__(self, data_dir):
        self.trainids, self.sampids, self.trainregs, self.sampleregs, self.regfeas = self.load_reg(data_dir)
        self.rel2id, self.kg_data, self.train_ent2id, self.trainkg_data, self.sample_ent2id, self.samplekg_data = self.load_kg(data_dir)
        self.train_data, self.nreg, self.min_data, self.max_data = self.load_flow(data_dir)  # ndays*nreg*nhour*2
        self.features, self.scale, self.scale_pred_data, self.scale_pred_X, self.KGE_pretrain = self.load_pretrain(data_dir)

        print('number of node=%d, number of edge=%d, number of relations=%d' % (self.nreg, len(self.kg_data), len(self.rel2id)))
        print('region num={}'.format(self.nreg))
        print('train data={}'.format(len(self.train_data)))
        print('load finished..')

    # Region
    def load_reg(self, data_dir):
        # ORI:
        # with open(data_dir + 'region2info.json', 'r') as f:
        #     region2info = json.load(f)
        # regions = sorted(region2info.keys(), key=lambda x: x)
        # reg2id = dict([(x, i) for i, x in enumerate(regions)])
        with open(data_dir + 'train_regs.json', 'r') as f:
            trainregs = json.load(f)
        with open(data_dir + 'test_regs.json', 'r') as f:
            testregs = json.load(f)
        sampregs = testregs
        trainids = [x for x in trainregs]
        sampids = [x for x in testregs]
        # ORI:
        # regfeas = []
        # for r in regions:
        #     tmp = region2info[r]['feature']
        #     regfeas.append(tmp)
        # regfeas = np.array(regfeas, dtype=np.float64)
        # Load the data from the given file path
        data = pd.read_csv("data/data_florida/aggregated_florida_visits_with_feature.csv")
        # Extract the 'feature' column and convert it from string representation of lists to actual lists
        # We use literal_eval to safely evaluate the string as a Python list
        data['feature'] = data['feature'].apply(lambda x: list(literal_eval(x)))
        # Sort the data by 'item_id' to ensure the order
        data_sorted = data.sort_values(by='item_id')
        # Extract the features into a list of lists (2D list)
        regfeas = data_sorted['feature'].tolist()

        # ORI: return reg2id, trainids, sampids, trainregs, sampregs, regfeas.tolist()
        return trainids, sampids, trainregs, sampregs, regfeas

    # Knowledge Graph
    def load_kg(self, data_dir):
        rel2id = {}
        kg_data_str = []
        trainkg_str = []
        samplekg_str = []
        with open(data_dir + 'kg.txt', 'r') as f:
            for line in f.readlines(): 
                h, r, t = line.strip().split('\t')
                kg_data_str.append((h, r, t))
                # train/sample kg
                if h in self.trainregs and t in self.trainregs:
                    trainkg_str.append((h, r, t))
                if h in self.sampleregs and t in self.sampleregs:
                    samplekg_str.append((h, r, t))
        ents = sorted(list(set([x[0] for x in kg_data_str] + [x[2] for x in kg_data_str])))
        rels = sorted(list(set([x[1] for x in kg_data_str])))
        for i, x in enumerate(ents):
            try:
                x
            except KeyError:
                x = self.nreg
        rel2id = dict([(x, i) for i, x in enumerate(rels)])
        kg_data = [[int(x[0]), rel2id[x[1]], int(x[2])] for x in kg_data_str]
        
        # train kg
        train_ent2id = dict([(x, i) for i, x in enumerate(self.trainregs)])
        trainkg_data = [[train_ent2id[x[0]], rel2id[x[1]], train_ent2id[x[2]]] for x in trainkg_str]
        # sample kg
        sample_ent2id = dict([(x, i) for i, x in enumerate(self.sampleregs)])
        samplekg_data = [[sample_ent2id[x[0]], rel2id[x[1]], sample_ent2id[x[2]]] for x in samplekg_str]

        return rel2id, kg_data, train_ent2id, trainkg_data, sample_ent2id, samplekg_data

    # alldayflow.json: day*nreg*nhour(24)*2
    def load_flow(self, data_dir):
        # def create_3d_visit_list(file_path):
        #     """
        #     Creates a 3D list from the dataframe with the following dimensions:
        #     1st Dimension: Based on unique 'bs' values.
        #     2nd Dimension: Sorted based on 'item_id' within each 'bs'.
        #     3rd Dimension: Monthly visit data from January 2019 to December 2019.
        #     """
        #     # Filter the dataframe for the year 2019
        #     df = pd.read_csv(file_path)
        #     columns_of_interest = ['region_id', 'bs', 'item_id'] + [f'2019-{month:02d}' for month in range(1, 13)]
        #     filtered_df = df[columns_of_interest]
        #
        #     # Initialize the 3D list
        #     visit_list_3d = []
        #
        #     # Iterate over each unique 'bs'
        #     for bs in filtered_df['bs'].unique():
        #         bs_data = filtered_df[filtered_df['bs'] == bs]
        #
        #         # Sort 'bs_data' by 'item_id' and iterate over each 'item_id'
        #         bs_list_2d = []
        #         for _, row in bs_data.sort_values(by='item_id').iterrows():
        #             # Extract the monthly visit data and append to the 2D list
        #             monthly_visits = row[3:].tolist()
        #             bs_list_2d.append(monthly_visits)
        #
        #         # Append the 2D list to the 3D list
        #         visit_list_3d.append(bs_list_2d)
        #
        #     return visit_list_3d

        def create_three_dimensional_visit_list(file_path):
            # Load the data
            data = pd.read_csv(file_path)
            # Select the relevant columns for visits
            columns_of_interest = ['bs', 'item_id'] + [f'2019-{month:02d}' for month in range(9, 13)]
            data = data[columns_of_interest]
            # Group by 'bs' and process each group
            grouped_data = data.groupby('bs')
            # Initialize the outer list (for each 'bs')
            outer_list = []
            for bs, group in grouped_data:
                # Sort the group by 'item_id'
                group_sorted = group.sort_values('item_id')
                # Extract only the monthly visit data as a list of lists (excluding 'bs' and 'item_id')
                inner_lists = group_sorted.iloc[:, 2:].values.tolist()
                # Append to the outer list
                outer_list.append(inner_lists)
            # Find the max length of the inner lists across all 'bs' groups
            max_length = max(len(inner_list) for inner_list in outer_list)
            # Ensure each inner list is of equal length, filling with mean values if necessary
            for inner_list in outer_list:
                while len(inner_list) < max_length:
                    # Calculate mean values for each month across existing items
                    mean_values = [np.nanmean([items[i] for items in inner_list]) for i in range(12)]
                    # Insert a new item with the mean values
                    inner_list.append(mean_values)
            return outer_list, np.array(outer_list).shape[1]

        def to_four_dimensions(three_dim_list):
            # 通过列表推导式遍历每个元素，并将其转换成一个新的列表
            return [[[[element] for element in inner_list] for inner_list in outer_list] for outer_list in
                    three_dim_list]

        # with open(data_dir + 'alldayflow.json', 'r') as f:
        #     date2flowmat = json.load(f)
        # train_data = []
        # for k, v in date2flowmat.items():
        #     if is_weekday(k):
        #         train_data.append(v)
        # train_data = np.array(train_data)
        # M, m = np.max(train_data), np.min(train_data)
        # train_data = (2 * train_data - m - M) / (M - m)  # 归一化到 [-1, 1]
        # return train_data.tolist(), m, M
        #
        # data = pd.read_csv(data_dir+"Florida_visits_2019_2020.csv")
        # filtered_data = data[(data['latitude'] >= min_lat) & (data['latitude'] <= max_lat) &
        #                      (data['longitude'] >= min_lon) & (data['longitude'] <= max_lon)]
        # visit_data_columns = filtered_data.columns[6:]  # 提取从2019年1月到2020年12月的访问数据列
        # visit_data = data[visit_data_columns].values.tolist()  # 将访问数据转换为二维列表

        visit_data, socall_nreg = create_three_dimensional_visit_list("data/data_florida/aggregated_florida_visits.csv")
        visit_data = to_four_dimensions(visit_data)
        train_data = np.array(visit_data)
        M, m = np.max(train_data), np.min(train_data)
        train_data = (2 * train_data - m - M) / (M - m)  # 归一化到 [-1, 1]
        return train_data.tolist(), socall_nreg, m, M

    def load_pretrain(self, data_dir):
        data = np.load(data_dir+'ER.npz')
        KGE_pretrain = data['E_pretrain']  # nreg*kgedim

        scale = np.array(self.train_data)  # nday*nreg*nhour*2
        scale = np.mean(scale, axis=0)  # nreg*nhour*2 计算每一列到均值
        scale = scale.reshape(self.nreg, -1)  # nreg*(nhour*2)
        scale = np.mean(scale, axis=1, keepdims=1)  # nreg*1
        
        scale_pred_data = []
        scale_pred_X = []
        for i in range(self.nreg):
            X = self.regfeas[i]
            scale_pred_data.append([X, scale[i].item()])
            scale_pred_X.append(X)
        features = np.array(scale_pred_X)

        return features, scale, scale_pred_data, scale_pred_X, KGE_pretrain
        