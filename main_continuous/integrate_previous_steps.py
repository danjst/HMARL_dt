import pandas as pd
import numpy as np
import setting
import math
import os
import setting

data_dir = '../data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def get_new_state_value(data, var_name, gamma):
    ratio_0 = (gamma**2)/((gamma**2) + gamma +1)
    ratio_1 = gamma/((gamma**2) + gamma +1)
    ratio_2 = 1/((gamma**2) + gamma +1)
    
    data['var_0'] = data[var_name]
    data['var_1'] = data.groupby('stay_id')[var_name].shift(1)
    data['var_2'] = data.groupby('stay_id')[var_name].shift(2)
    
    data['var_1'] = data['var_1'].fillna(data['var_0'])
    data['var_2'] = data['var_2'].fillna(data['var_1'])
    
    data['context_' + var_name] = data['var_0']*ratio_0 + data['var_1']*ratio_1 + data['var_2']*ratio_2
    data = data.drop(['var_0', 'var_1','var_2'], axis=1)

    return data


if __name__ == "__main__":
    df = pd.read_csv('../data/data_rl_4h_train_test_split_3steps_siqi.csv')

    df.fillna(0, inplace=True)

    for var_name in setting.state_col:
        df = get_new_state_value(df, var_name, math.e)

    for next_var_name in setting.next_state_col:
        df = get_new_state_value(df, next_var_name, math.e)


    df.to_csv('../data/context.csv', index= False)

