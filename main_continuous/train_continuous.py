"""Trains off-policy algorithms, such as QMIX and IQL."""

import json
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['TF_CUDNN_DETERMINISTIC']='1'
import random
import sys
import time
import math
import pandas as pd
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import embedding_alg as alg_qmix
from tqdm import tqdm
import logging
import argparse
import setting
from embedding_evaluation_graphs import *
import embedding_networks as networks


state_col = setting.state_col
next_state_col = setting.next_state_col
action_dis_col = setting.action_dis_col
action_con_col = setting.action_con_col
ai_action_con_col = setting.ai_action_con_col
ITERATION_ROUND_PRETRAIN = setting.ITERATION_ROUND_PRETRAIN
ITERATION_ROUND_QMIX = setting.ITERATION_ROUND_QMIX
ITERATION_ROUND_Vaso = setting.ITERATION_ROUND_Vaso
ITERATION_ROUND_IV = setting.ITERATION_ROUND_IV
ITERATION_ROUND = setting.ITERATION_ROUND
TOTAL_ITERATION_NUM = setting.TOTAL_ITERATION_NUM
ACTION_SPACE = setting.ACTION_SPACE
BATCH_SIZE = setting.BATCH_SIZE
STATE_DIM = len(state_col) #48 
REWARD_FUN = setting.REWARD_FUN
context_state_col = setting.context_state_col
context_next_state_col = setting.context_next_state_col
hidden_factor = setting.hidden_factor
l_state = len(setting.state_col)
nn = setting.nn
Q_threshold = setting.Q_threshold

FM_list = ['FM_' + str(i) for i in range(setting.hidden_factor)]
next_FM_list = ['next_FM_' + str(i) for i in range(setting.hidden_factor)]
FM_context_list = ['FM_' + str(i) for i in range(setting.hidden_factor)]
FM_context_list.extend(['FM_context' + str(i) for i in range(setting.hidden_factor)])
next_FM_context_list = ['next_FM_' + str(i) for i in range(setting.hidden_factor)]
next_FM_context_list.extend(['next_FM_context' + str(i) for i in range(setting.hidden_factor)])


intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),)

def display_time(seconds, granularity=2):
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])

def compute_master_action(iv_fluids_quantile, vasopressors_quantile):
    if iv_fluids_quantile==1 and vasopressors_quantile==1:
        master_action=0
    elif iv_fluids_quantile==1 and vasopressors_quantile>1:
        master_action=2
    elif iv_fluids_quantile>1 and vasopressors_quantile==1:
        master_action=1
    else:
        master_action = 3
    return master_action

def convert_to_unit(old_value,old_max, old_min, new_max, new_min):
    new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    return new_value

def get_new_state_value(data, var_name, gamma):
    ratio_0 = (gamma**2)/((gamma**2) + gamma +1)
    ratio_1 = gamma/((gamma**2) + gamma +1)
    ratio_2 = 1/((gamma**2) + gamma +1)
    
    data['var_0'] = data[var_name]
    data['var_1'] = data.groupby('stay_id')[var_name].shift(1)
    data['var_2'] = data.groupby('stay_id')[var_name].shift(2)
    
    data['var_1'] = data['var_1'].fillna(df['var_0'])
    data['var_2'] = data['var_2'].fillna(df['var_1'])
    
    data[var_name] = data['var_0']*ratio_0 + data['var_1']*ratio_1 + data['var_2']*ratio_2
    data = data.drop(['var_0', 'var_1','var_2'], axis=1)

    return data

def pre_train_master_RL(RL, data, first_run=True):
    if first_run:
         # reward function
        data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)        
        memory_array = np.concatenate([np.array(data[state_col]),  # index 0:47
                            np.array(data['master_action']).reshape(-1, 1), # index 48, master action made in "__main__"
                            np.array(data['reward']).reshape(-1, 1), # index 49, rewards created from 3 line above
                            np.array(data['done']).reshape(-1, 1), # index 50, done column already created in data.csv
                            np.array(data[next_state_col])], #index 51:98
                            axis = 1)
        np.save('../data/continuous/pretrain_master_memory.npy', memory_array)
        
    else:
        
        memory_array = np.load('../data/continuous/pretrain_master_memory.npy')

    print('\nSTART store_transition\n')
    RL.store_transition(memory_array)
    
    print('\nSTART PRE-TRAINING MASTER AGENT\n')

#     EPISODE = int(MEMORY_SIZE / BATCH_SIZE * ITERATION_ROUND_PRETRAIN)
  
    for i in tqdm(range(ITERATION_ROUND_PRETRAIN)):
        RL.learn(i, pretrain = True)
    loss = RL.cost_his
    return loss


def train_master_RL(RL, data, use_FM, first_run=True):
    if first_run:
         # reward function
        data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)        
        memory_array = np.concatenate([np.array(data[state_col]),
                                       np.array(data[next_state_col]),
                                       np.array(data['master_action']).reshape(-1, 1),
                                       np.array(data['reward']).reshape(-1, 1), 
                                       np.array(data['done']).reshape(-1, 1),
                                       np.array(data['Q_phys_no_action']).reshape(-1,1),
                                       np.array(data['Q_phys_IV_only']).reshape(-1,1),
                                       np.array(data['Q_phys_Vasso_only']).reshape(-1,1),
                                       np.array(data['Q_phys_qmix']).reshape(-1,1),
                                       np.array(data[FM_context_list]),
                                       np.array(data[next_FM_context_list])],
                                       axis = 1)
        np.save('../data/continuous/master_memory.npy', memory_array)
        
    else:
        
        memory_array = np.load('../data/continuous/master_memory.npy', allow_pickle=True)

    print('\nSTART store_transition\n')
    RL.store_transition(memory_array)
    
    print('\nSTART TRAINING MASTER AGENT\n')

    if use_FM>0:
        print('\nSTART TRAINING MASTER AGENT with Embedding K{}\n'.format(setting.hidden_factor))
    else:
        print('\nSTART TRAINING MASTER AGENT WITH NO EMBEDDING\n')

  
    for i in tqdm(range(ITERATION_ROUND)):
        RL.learn(i, use_FM, pretrain = False)
    loss = RL.cost_his
    return loss

def train_single_RL_IV(RL, data, use_FM, first_run=True, writer = None, epoch = 1):
    if first_run:
         # reward function
        data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)        
        memory_array = np.concatenate([np.array(data[state_col]), 
                                       np.array(data[next_state_col]), 
                                       np.array(data[action_con_col[0]]).reshape(-1,1), 
                                       np.array(data['reward']).reshape(-1, 1), 
                                       np.array(data['done']).reshape(-1, 1),
                                       np.array(data[FM_context_list]), 
                                       np.array(data[next_FM_context_list])],
                                       axis = 1)
        np.save('../data/continuous/IV_only_memory.npy', memory_array)
    else:
        memory_array = np.load('../data/continuous/IV_only_memory.npy', allow_pickle=True)

    print('\nSTART store_transition\n')
    RL.store_transition(memory_array)
    

    if use_FM>0:
        print('\nSTART TRAINING IV AGENT with Embedding K{}\n'.format(setting.hidden_factor))
    else:
        print('\nSTART TRAINING IV AGENT WITH NO EMBEDDING\n')

    EPISODE = int(ITERATION_ROUND_IV)
    for i in tqdm(range(EPISODE)):
        RL.train_step_single_AC(i, use_FM, writer = writer, iv_action_only = True)
        

    IV_loss = RL.cost_his
    return IV_loss

def train_single_RL_Vasso(RL, data, use_FM, first_run=True,writer = None, epoch = 1):
    if first_run:
         # reward function
        data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)
        memory_array = np.concatenate([np.array(data[state_col]),
                                       np.array(data[next_state_col]),
                                       np.array(data[action_con_col[1]]).reshape(-1,1),
                                       np.array(data['reward']).reshape(-1, 1),
                                       np.array(data['done']).reshape(-1, 1),
                                       np.array(data[FM_context_list]),
                                       np.array(data[next_FM_context_list])],
                                       axis = 1)
        np.save('../data/continuous/vasso_only_memory.npy', memory_array)
        
    else:
        
        memory_array = np.load('../data/continuous/vasso_only_memory.npy', allow_pickle=True)

    print('\nSTART store_transition\n')
    RL.store_transition(memory_array)
    
    if use_FM>0:
        print('\nSTART TRAINING VASO AGENT with Embedding K{}\n'.format(setting.hidden_factor))
    else:
        print('\nSTART TRAINING VASO AGENT WITH NO EMBEDDING\n')
    EPISODE = int(ITERATION_ROUND_Vaso)
    for i in tqdm(range(EPISODE)):
        RL.train_step_single_AC(i,use_FM, writer = writer, iv_action_only= False)
        

    Vasso_loss = RL.cost_his
    return Vasso_loss

def train_mixer(RL, data, use_FM, first_run=True, writer = None, epoch = 1):
    if first_run:
         # reward function
        data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)  
        memory_array = np.concatenate([np.array(data[state_col]), # 48
                                       np.array(data[next_state_col]), # 48
                                       np.array(data[action_con_col]),  # 2
                                       np.array(data[ai_action_con_col]), # 2
                                       np.array(data['reward']).reshape(-1, 1),  #1
                                       np.array(data['done']).reshape(-1, 1), # 1
                                       np.array(data[FM_context_list]), # 16 FM(size=8) + FM_context(size=8)
                                       np.array(data[next_FM_context_list])],axis = 1) # 16 FM + FM context
        np.save('../data/continuous/hierarchy_discrete_memory.npy', memory_array) 
        
    else:
        
        memory_array = np.load('../data/continuous/Qmix_discrete_memory.npy')

    print('\nSTART store_transition\n')
    RL.store_transition(memory_array)
    
    if use_FM>0:
        print('\nSTART TRAINING QMIX AGENT with Embedding K{}\n'.format(setting.hidden_factor))
    else:
        print('\nSTART TRAINING QMIX AGENT WITH NO EMBEDDING\n')
    MEMORY_SIZE_mix = len(data)

#     EPISODE = int(MEMORY_SIZE_mix / BATCH_SIZE * ITERATION_ROUND_QMIX)
    EPISODE = int(ITERATION_ROUND_QMIX)
    for i in tqdm(range(EPISODE)):
        RL.train_step(i, use_FM, writer = writer)
        

    loss = RL.cost_his
    return loss

def train_function(df, use_FM, train_FM):
    
    start_time = time.time()
    
    seed = setting.SEED
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    summarize = False
    
    
    l_state = STATE_DIM
    l_action = 2   
    N_agent = 2
    master_action_num = setting.master_action_num

    data = df[df['train_test']=="train"] # daniel   
    data =data.reset_index(drop = True) #daniel

    if use_FM>0:
        input_dim = 2*setting.hidden_factor
    else:
        input_dim = STATE_DIM
# ############## pre-train master agent #####################

    if (train_FM>0):
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        sess = tf.Session(config=config_proto)

        RL_master = alg_qmix.DuelingDQN(n_actions=master_action_num, n_features=STATE_DIM, memory_size=len(data),
                                       batch_size=BATCH_SIZE, e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=True, pretrain = True, K_hidden_factor= setting.hidden_factor)

        sess.run(tf.global_variables_initializer())

        pre_master_loss = pre_train_master_RL(RL_master, data, first_run=True)
        # save model
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, 'models/pretrain/duel_DQN')
        new_saver = tf.compat.v1.train.import_meta_graph('models/pretrain/duel_DQN.meta')
        new_saver.restore(sess, 'models/pretrain/duel_DQN')
    # #########################################################################    
    # embedding creator RL_master model has already been trained fully starting at this point
        # evaluate model
        eval_batch_size = int(20000)
        index_num_0 = int(0)
        index_num_1 = int(eval_batch_size)
        eval_result = pd.DataFrame(columns=list(FM_context_list) + list(next_FM_context_list)+['Q_phys_no_action'])
        while (index_num_1 < len(df)):
            eval_batch = df.iloc[index_num_0:index_num_1]
            eval_q, state_embedding = sess.run([RL_master.q_eval, RL_master.Em_State],
                                               feed_dict={RL_master.s: eval_batch[state_col]})

            next_state_embedding = sess.run([RL_master.Em_State],
                                            feed_dict={RL_master.s: eval_batch[next_state_col]})
            context_embedding = sess.run([RL_master.Em_State],
                                    feed_dict={RL_master.s: eval_batch[context_state_col]})
            next_context_embedding = sess.run([RL_master.Em_State],
                                            feed_dict={RL_master.s: eval_batch[context_next_state_col]})

            state_embedding = np.array(state_embedding).reshape(-1, hidden_factor)
            next_state_embedding = np.array(next_state_embedding).reshape(-1,hidden_factor)
            context_embedding = np.array(context_embedding).reshape(-1,hidden_factor)
            next_context_embedding = np.array(next_context_embedding).reshape(-1, hidden_factor)
            Q_no = eval_q[:,0].reshape(-1,1) # [:,0] corresponds to the q-value of 0th action (no-action), 1 is iv, 2 is vaso, 3 is mix 

            model_result = np.concatenate([state_embedding,context_embedding, next_state_embedding, next_context_embedding,Q_no], axis=1)
 
            model_result = pd.DataFrame(model_result, columns =list(eval_result.columns))
            #model_result.to_csv('../data/model_result.csv', index = False)
            #eval_result.to_csv('../data/eval_result.csv', index = False)
            eval_result = pd.concat([eval_result, model_result], ignore_index=True)
            index_num_0 = index_num_1
            index_num_1 = index_num_1 +eval_batch_size
        print(index_num_0)
        eval_batch = df.iloc[index_num_0:]
        eval_q, state_embedding = sess.run([RL_master.q_eval, RL_master.Em_State],
                                           feed_dict={RL_master.s: eval_batch[state_col]})
        next_state_embedding = sess.run([RL_master.Em_State],
                                        feed_dict={RL_master.s: eval_batch[next_state_col]})
        context_embedding = sess.run([RL_master.Em_State],
                                feed_dict={RL_master.s: eval_batch[context_state_col]})
        next_context_embedding = sess.run([RL_master.Em_State],
                                        feed_dict={RL_master.s: eval_batch[context_next_state_col]}) 
        state_embedding = np.array(state_embedding).reshape(-1, hidden_factor)
        next_state_embedding = np.array(next_state_embedding).reshape(-1,hidden_factor)
        context_embedding = np.array(context_embedding).reshape(-1,hidden_factor)
        next_context_embedding = np.array(next_context_embedding).reshape(-1, hidden_factor)
        Q_no = eval_q[:,0].reshape(-1,1)  # [:,0] corresponds to the q-value of 0th action (no-action), 1 is iv, 2 is vaso, 3 is mix 
        model_result = np.concatenate([state_embedding,context_embedding, next_state_embedding, next_context_embedding,Q_no], axis=1)
        model_result = pd.DataFrame(model_result, columns =list(eval_result.columns))
        eval_result = pd.concat([eval_result, model_result], ignore_index=True)

        result_array = np.concatenate([df.values, eval_result.values], axis=1)
        result = pd.DataFrame(result_array, 
                              columns=list(df.columns)+list(eval_result.columns))


        print(result.head(1))
        print(len(result))
        result.to_csv('../data/mimic_embeddings_K'+str(setting.hidden_factor)+'_Itr'+str(setting.ITERATION_ROUND_PRETRAIN)+'.csv', encoding = 'gb18030', index = False)
        df = result.copy()        
        df['Q_phys_no_action'] = df['Q_phys_no_action'].apply(lambda x: -Q_threshold if x<-Q_threshold else x if x<Q_threshold else Q_threshold)

    else:
        print("loading data...")
        df = pd.read_csv(('../data/mimic_embeddings_K'+str(setting.hidden_factor)+'_Itr'+str(setting.ITERATION_ROUND_PRETRAIN)+'.csv'), index_col = False)
        df['master_action'] = df.apply(lambda x: compute_master_action(x['iv_fluids_quantile'], x['vasopressors_quantile']), axis=1)

        df['Q_phys_no_action'] = df['Q_phys_no_action'].apply(lambda x: -Q_threshold if x<-Q_threshold else x if x<Q_threshold else Q_threshold)









      
 ################### single_AC network for IV only ################### 
    IV_only_data = df[(df['master_action']==1) & (df['train_test']=="train")]
    IV_only_data = IV_only_data.reset_index(drop=True)

    #IV_only_data.to_csv('../data/iv_only_data.csv',index=False)
    
    tf.reset_default_graph()
    alg = alg_qmix.Single_AC(1, l_state, hidden_factor, input_dim, l_action, nn, 4e-4, 2e-4, memory_size=len(IV_only_data), batch_size=BATCH_SIZE, e_greedy_increment=0.001) 


    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    sess.run(alg.list_initialize_target_ops)
    writer = tf.compat.v1.summary.FileWriter('models/IV_only', sess.graph)
    saver = tf.compat.v1.train.Saver(max_to_keep=100) 
    
    
    iv_loss = train_single_RL_IV(alg, IV_only_data, use_FM, first_run=True, writer = writer,epoch = 1)
    # save model
    saver.save(sess, 'models/IV_only/con_IV_only_model.ckpt')

    i = 2
    while(i<=TOTAL_ITERATION_NUM):
        new_saver = tf.compat.v1.train.import_meta_graph('models/IV_only/con_IV_only_model.ckpt.meta')
        new_saver.restore(sess, 'models/IV_only/con_IV_only_model.ckpt')
        iv_loss = train_single_RL_IV(alg, IV_only_data, use_FM, first_run=False, writer = writer, epoch = i)
        # save model
        new_saver.save(sess, 'models/IV_only/con_IV_only_model.ckpt')
        i = i+1 


    
    # evaluate single IV model (WHOLE DATASET including all master_action = 1,2,3,4)
    if (use_FM>0):
        eval_state = np.array(df[FM_context_list]) # now using whole DF instead of IV-active only rows, so now including rows where IV value is 0
        eval_obs = np.array(eval_state).reshape((-1,2*setting.hidden_factor))
    else:
        eval_state = np.array(df[state_col])
        eval_obs = np.array(eval_state).reshape((-1,len(state_col)))    

    
    
    actions_int = alg.run_actor(eval_obs, sess) # ai actions
    a_0 = df[action_con_col[0]]   # physician actions
    phys_qmix = alg.run_phys_Q_continuous(sess, list_state=eval_state, list_obs = eval_obs, a_0=a_0) # get q value physician actions (1 value which is for IV continuous)
    ai_qmix = alg.run_RL_Q_continuous(sess, list_state=eval_state, list_obs = eval_obs, a_0=actions_int[:,0]) # get q value for ai model actions (1 value which is for IV continuous)
    
    result_array = np.concatenate([df.values, actions_int, phys_qmix,ai_qmix], axis=1)
    result = pd.DataFrame(result_array, 
                          columns=list(df.columns)+['ai_action_IV_only','Q_phys_IV_only','Q_ai_IV_only'])
    print("result")
    print(result.head(1))

    #result.to_csv('../data/iv_results.csv',index=False)

################### single_AC network for Vasso only ################### 

    df = result.copy()
    temp = df[df['train_test']=="train"]
    Vasso_only_data = temp[temp['master_action'].isin([2,3]) ]
    Vasso_only_data = Vasso_only_data.reset_index(drop=True)
    
    tf.reset_default_graph()
    alg = alg_qmix.Single_AC(1, l_state, hidden_factor, input_dim, l_action, nn,1e-3, 2e-4 , memory_size=len(Vasso_only_data), batch_size=BATCH_SIZE, e_greedy_increment=0.001)    

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    sess.run(alg.list_initialize_target_ops)
    writer = tf.compat.v1.summary.FileWriter('models/Vasso_only/', sess.graph)
    saver = tf.compat.v1.train.Saver(max_to_keep=100)
    
    
    vasso_loss = train_single_RL_Vasso(alg, Vasso_only_data, use_FM, first_run=True, writer = writer, epoch = 1)
    # save model
    saver.save(sess, 'models/Vasso_only/con_Vasso_only_model.ckpt')
    i = 2
    while(i<=TOTAL_ITERATION_NUM):
        new_saver = tf.compat.v1.train.import_meta_graph('models/Vasso_only/Vasso_only_model.ckpt.meta')
        new_saver.restore(sess, 'models/Vasso_only/con_Vasso_only_model.ckpt')
        vasso_loss = train_single_RL_Vasso(alg, Vasso_only_data, use_FM, first_run=True, writer = writer,epoch = i)
        # save model
        new_saver.save(sess, 'models/Vasso_only/con_Vasso_only_model.ckpt')
        i = i+1     
    
    
    # evaluate single vaso model
    
    if (use_FM>0):
        eval_state = np.array(df[FM_context_list])
        eval_obs = np.array(eval_state).reshape((-1,2*setting.hidden_factor))
    else:
        eval_state = np.array(df[state_col])
        eval_obs = np.array(eval_state).reshape((-1,len(state_col)))    

        

    actions_int = alg.run_actor(eval_obs, sess)
    a_0 = df[action_con_col[1]]
    phys_qmix = alg.run_phys_Q_continuous(sess, list_state=eval_state, list_obs = eval_obs, a_0=a_0)
    ai_qmix = alg.run_RL_Q_continuous(sess, list_state=eval_state, list_obs = eval_obs, a_0=actions_int[:,0])
    
    result_array = np.concatenate([df.values, actions_int, phys_qmix,ai_qmix], axis=1)
    result = pd.DataFrame(result_array, 
                          columns=list(df.columns)+['ai_action_Vasso_only','Q_phys_Vasso_only','Q_ai_Vasso_only'])


    print("result")
    print(result.head(1))

    #result.to_csv('../data/vaso_results.csv',index=False)

# ################################ Qmix network ###############################

    df = result.copy()
    mixer_data = df[(df['master_action']==3) & (df['train_test']=="train")]
    mixer_data = mixer_data.reset_index(drop = True)
    
    tf.reset_default_graph()

    alg = alg_qmix.Qmix(N_agent, l_state, hidden_factor, input_dim,l_action,nn, memory_size=len(mixer_data), batch_size=BATCH_SIZE, e_greedy_increment=0.001)    

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    sess.run(alg.list_initialize_target_ops)
    writer = tf.compat.v1.summary.FileWriter('models/qmix/', sess.graph)
    saver_mixer = tf.compat.v1.train.Saver(max_to_keep=100)
    
    
    mixer_loss = train_mixer(alg, mixer_data, use_FM, first_run=True, writer = writer, epoch = 1)
    # save model
    saver_mixer.save(sess, 'models/qmix/con_qmix_model.ckpt')
    i = 2
    while(i<=TOTAL_ITERATION_NUM):
        new_saver = tf.compat.v1.train.import_meta_graph('models/qmix/con_qmix_model.ckpt.meta')
        new_saver.restore(sess, 'models/qmix/con_qmix_model.ckpt')
        mixer_loss = train_mixer(alg, mixer_data, use_FM, first_run=True, writer = writer,epoch = i)
        # save model
        new_saver.save(sess, 'models/qmix/con_qmix_model.ckpt')
        i = i+1      
    # evaluate model
    if (use_FM>0):
        eval_state = np.array(df[FM_context_list])
        eval_obs = np.stack((eval_state, eval_state)).reshape((-1,2*setting.hidden_factor))

    else:
        eval_state = np.array(df[state_col])
        eval_obs = np.stack((eval_state, eval_state)).reshape((-1,len(state_col)))    

    a_0 = df[action_con_col[0]]
    a_1 = df[action_con_col[1]]
    iv_only = df[ai_action_con_col[0]]
    vasso_only = df[ai_action_con_col[1]]
    iv_actions, vaso_actions = alg.run_actor(eval_obs, sess, iv_only = iv_only, vasso_only = vasso_only)    
    phys_qmix = alg.run_phys_Q_continuous(sess, list_state=eval_state, list_obs = eval_obs, a_0=a_0, a_1=a_1, iv_only = iv_only, vasso_only = vasso_only)
    ai_qmix = alg.run_RL_Q_continuous(sess, list_state=eval_state, list_obs = eval_obs, a_0=iv_actions, a_1=vaso_actions, iv_only = iv_only, vasso_only = vasso_only)
    
    result_array = np.concatenate([df.values, iv_actions, vaso_actions, phys_qmix,ai_qmix], axis=1)
    result = pd.DataFrame(result_array, 
                          columns=list(df.columns)+['ai_action_qmix_IV', 'ai_action_qmix_Vasso','Q_phys_qmix','Q_ai_qmix'])
    if use_FM>0:
        res_dir = '../data/con_Qmix_result_K'+str(setting.hidden_factor)+'.csv'
        result.to_csv(res_dir, encoding = 'gb18030')
    else:
        result.to_csv('../data/con_Qmix_result_NoFM.csv', encoding = 'gb18030')


################### Master Agent to decide no_action, IV_only, Vasso_only, or Qmix ###################
    tf.reset_default_graph()
    df = result.copy()
    combined_data = df[df['train_test']=="train"]
    combined_data = combined_data.reset_index(drop = True)


    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)

    RL_master = alg_qmix.MasterDQN(master_action_num, STATE_DIM, hidden_factor, input_dim,
                                   memory_size=len(combined_data),
                                   batch_size=BATCH_SIZE, e_greedy_increment=0.001, sess=sess, 
                                   dueling=True, output_graph=True, pretrain = False)
    
    sess.run(tf.global_variables_initializer())

    master_loss = train_master_RL(RL_master, combined_data, use_FM,first_run=True)
    # save model
    saver = tf.compat.v1.train.Saver()
    saver.save(sess, 'models/master/con_duel_DQN')

    

#     evaluate master agent
    if (use_FM>0):
        eval_state = np.array(df[FM_context_list])
        eval_obs = np.array(eval_state).reshape((-1,2*setting.hidden_factor))

    else:
        eval_state = np.array(df[state_col])
        eval_obs = np.array(eval_state).reshape((-1,len(state_col))) 
  
   
    eval_q = sess.run(RL_master.q_eval, feed_dict={RL_master.s:eval_state})    
    result_array = np.concatenate([df.values, eval_q], axis=1)
    result = pd.DataFrame(result_array, 
                          columns=list(df.columns)+['Q_0', 'Q_1', 'Q_2', 'Q_3'])
    

    Q_list = ['Q_' + str(i) for i in range(master_action_num)]
    result['Q_ai_master'] = np.max(result[Q_list],axis = 1)
    result['ai_action_master'] = np.argmax(np.array(result[Q_list]),axis = 1)
    print("result")
    print(result.head(1))
    if use_FM>0:
        res_dir = '../data/con_main_result_K'+str(setting.hidden_factor)+'_Itr'+str(setting.ITERATION_ROUND)+'.csv'
        result.to_csv(res_dir, encoding = 'gb18030')
    else:
        result.to_csv('../data/con_main_result_NoFM.csv', encoding = 'gb18030')


 
    run_time = display_time((time.time()-start_time))
    print("done!")   
    print("Total run time with {} episodes:\n {}".format(setting.ITERATION_ROUND, run_time))    
    print("start evaluation")
    train_result = result[result['train_test']=="train"]
    train_result = train_result.reset_index(drop=True)
    test_result = result[result['train_test']=="test"]
    test_result = test_result.reset_index(drop =True)    
    if train_FM==0:
        pre_master_loss = 0    
    if use_FM>0:
        print("evaluating train result")
        run_eval(train_result, pre_master_loss, master_loss, iv_loss, vasso_loss, mixer_loss, datatype = 'mimic', phase = "train_Embedding", train_FM = train_FM)
        print("evaluating test result")
        run_eval(test_result, pre_master_loss, master_loss, iv_loss, vasso_loss, mixer_loss, datatype = 'mimic', phase = "test_Embedding", train_FM = train_FM)
    else:
        print("evaluating train result")
        run_eval(train_result, pre_master_loss, master_loss, iv_loss, vasso_loss, mixer_loss, datatype = 'mimic', phase = "train", train_FM=train_FM)
        print("evaluating test result")
        run_eval(test_result, pre_master_loss, master_loss, iv_loss, vasso_loss, mixer_loss, datatype = 'mimic', phase = "test", train_FM=train_FM)    


# ################### MAIN #########################    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_FM', '-e', type=float, required=True)
    parser.add_argument('--retrain_FM', '-train_FM', type=float, required=True)
    args = parser.parse_args()  

    #df = pd.read_csv('../data/data_rl_4h_train_test_split_3steps.csv')  #daniel
    df = pd.read_csv('../data/context.csv')  #daniel

    df.fillna(0, inplace=True) #daniel
    df['master_action'] = df.apply(lambda x: compute_master_action(x['iv_fluids_quantile'], x['vasopressors_quantile']), axis=1) #daniel

    train_function(df,args.use_FM, args.retrain_FM)        
