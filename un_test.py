#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:08:13 2020

@author: anubhav
"""
import json
import os
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
import torch
import random
import numpy as np

import pickle


# pk_old = pickle.load(open('/home/anubhav/saarthi.ai/madpl-multiwoz/data/processed_data/goal_model.pkl', 'rb'))

pk = open("dstc-data/processed_data/goal_model_dstc.pkl","rb")
goal_model = pickle.load(pk)

def nomial_sample(counter: Counter):
    return list(counter.keys())[np.argmax(np.random.multinomial(1, list(counter.values())))]

domain = 'Restaurants_2'
cnt_slot = goal_model[0][domain]
cnt_slot_value = goal_model[1][domain]
domain_goal = {'info':{}}

for slot in cnt_slot['info']:
    if slot in list(cnt_slot_value['info'].keys()):
        domain_goal['info'][slot] = nomial_sample(cnt_slot_value['info'][slot])
        
reqt = [slot for slot in cnt_slot['info'] if slot not in list(cnt_slot['reqt'].keys())]
print(reqt)
print(cnt_slot['reqt'])
for slot in cnt_slot['info']:
    if slot not in list(cnt_slot['reqt'].keys()):
        reqt.append(slot)
if len(reqt) > 0:
    domain_goal['reqt']= reqt
    
        
        
        
# services = {'RentalCars_2', 'Movies_3', 'Services_2', 'Media_2', 'Restaurants_1', 'Homes_2', 'Payment_1', 'Calendar_1', 'Flights_1', 'Weather_1', 'Trains_1', 'Messaging_1', 'RentalCars_3', 'Music_2', 'Hotels_3', 'Buses_3', 'Music_3', 'Travel_1', 'Flights_3', 'Homes_1', 'Music_1', 'Alarm_1', 'Movies_2', 'Events_2', 'Media_1', 'RentalCars_1', 'Services_3', 'Buses_2', 'Flights_2', 'Hotels_4', 'Hotels_2', 'RideSharing_2', 'Events_3', 'Media_3', 'Services_4', 'Restaurants_2', 'Movies_1', 'Hotels_1', 'Buses_1', 'Banks_1', 'Events_1', 'Services_1', 'Flights_4', 'RideSharing_1', 'Banks_2'}

# typs = ['info', 'reqt', 'book']

# pk_extra = pickle.load(open('/home/anubhav/saarthi.ai/agreement/goal_model_dstc.pkl', 'rb'))
# with open('dstc-data/dstc_slot_types.json', 'rb') as f:
#     data = json.load(f)
    
    
    
# slot_cnt = {}
# slot_value = {}

# for item in services:
#     slot_cnt[item] = {}
#     slot_value[item] = {}

# for item in services:
#     for ty in typs:
#         slot_cnt[item][ty] = {}
#         slot_value[item][ty] = {}

# for doms in data[0].keys():
#     for typ in data[0][doms].keys():
#         sl = data[0][doms][typ]
#         ls = pk_extra[0][doms].keys()
#         for its in sl:
#             for itl in ls:
#                 if its == itl:
#                     slot_cnt[doms][typ][its] = pk_extra[0][doms][its]
                    
# for doms in data[0].keys():
#     for typ in data[0][doms].keys():
#         sl = data[0][doms][typ]
#         ls = pk_extra[1][doms].keys()
#         for its in sl:
#             for itl in ls:
#                 if its == itl:
#                     slot_value[doms][typ][its] = deepcopy(pk_extra[1][doms][its])
                    



        
        
        
        
                    

        # cnt_slot = self.ind_slot_dist[domain]
        # cnt_slot_value = self.ind_slot_value_dist[domain]
        
# domain = 'Restaurants_2'
# # ind_slot_dist = 0
# # ind_slot_value_dist = 1
# cnt_slot = goal_model[0][domain]
# cnt_slot_value = goal_model[1][domain]

# domain_goal = {}

# for slot in cnt_slot.keys():
#     r_no = random.random()
#     print(r_no)
#     print(slot)
#     if r_no < cnt_slot[slot] + 0.2:
#         domain_goal[slot] = nomial_sample(cnt_slot_value[slot])





















# services = {'RentalCars_2', 'Movies_3', 'Services_2', 'Media_2', 'Restaurants_1', 'Homes_2', 'Payment_1', 'Calendar_1', 'Flights_1', 'Weather_1', 'Trains_1', 'Messaging_1', 'RentalCars_3', 'Music_2', 'Hotels_3', 'Buses_3', 'Music_3', 'Travel_1', 'Flights_3', 'Homes_1', 'Music_1', 'Alarm_1', 'Movies_2', 'Events_2', 'Media_1', 'RentalCars_1', 'Services_3', 'Buses_2', 'Flights_2', 'Hotels_4', 'Hotels_2', 'RideSharing_2', 'Events_3', 'Media_3', 'Services_4', 'Restaurants_2', 'Movies_1', 'Hotels_1', 'Buses_1', 'Banks_1', 'Events_1', 'Services_1', 'Flights_4', 'RideSharing_1', 'Banks_2'}

# corpus_path = 'dstc-data/dstc_dev.json'
# goal_path = 'dstc-data/dstc_goals_dev.json'

# data_dir = 'dstc-data'
# goal_filename_train = 'dstc_goals_train.json'
# goal_path_train = data_dir + '/' + goal_filename_train
        
# goal_filename_dev = 'dstc_goals_dev.json'
# goal_path_dev = data_dir + '/' + goal_filename_dev
        
# goal_filename_test = 'dstc_goals_test.json'
# goal_path_test = data_dir + '/' + goal_filename_test

# # with open(goal_path_train) as fgt:
# #     goals_train = json.load(fgt)

# # with open(goal_path_test) as fgte:
# #     goals_test = json.load(fgte)
    
# with open(goal_path_dev) as fgd:
#     goals = json.load(fgd)
 
# # goals = []

# # goal_files = [goals_train, goals_test, goals_dev]
# # for ith in goal_files:
# #     for itm in ith:
# #         goals.append(itm)

# print('Calculating Individual Slot Value Distribution')
# ind_slot_value_cnt = dict([(domain, {}) for domain in services])
# for i in tqdm(goals):
#     for idx, g in enumerate(goals):
#         for domain in services:
#             if g['goals'][domain] != {}:
#                 for slot in g['goals'][domain].keys():
#                     if g['goals'][domain][slot] != []:
#                         ind_slot_value_cnt[domain][slot] = {}
#                         val = g['goals'][domain][slot]
#                         for item in val:
#                             if item not in list(ind_slot_value_cnt[domain][slot].keys()):
#                                 ind_slot_value_cnt[domain][slot][item] = 1
#                             else:
#                                 ind_slot_value_cnt[domain][slot][item] += 1

# # self.ind_slot_value_dist = deepcopy(ind_slot_value_cnt)
# ind_slot_value_dist = deepcopy(ind_slot_value_cnt)
# for itk in ind_slot_value_cnt.keys():
#     for itp in ind_slot_value_cnt[itk]:
#         for order in ind_slot_value_cnt[itk][itp]:
#             ind_slot_value_dist[itk][itp][order] = ind_slot_value_cnt[itk][itp][order] / sum(ind_slot_value_cnt[itk][itp].values())


# print('Calculating Domain Distribution')
# domain_orderings = []
# for k in tqdm(goals):
#     for g in goals:
#         d_domains = []
#         for it in g['goals'].keys():
#             if g['goals'][it] != {}:
#                 d_domains.append(it)
#         domain_orderings.append(tuple(d_domains))
# domain_orderings = list(domain_orderings)
    
# domain_ordering_cnt = Counter(domain_orderings)
# # self.domain_ordering_dist = deepcopy(domain_ordering_cnt)
# domain_ordering_dist = deepcopy(domain_ordering_cnt)

# for order in domain_ordering_cnt.keys():
#     # self.domain_ordering_dist[order] = domain_ordering_cnt[order] / sum(domain_ordering_cnt.values())
#     domain_ordering_dist[order] = domain_ordering_cnt[order] / sum(domain_ordering_cnt.values())

# # dialogs[d]['goal'][domain] is analogous  g['goals'][domain]

# print('Calculating Individual Slot Distribution')
# ind_slot_cnt = dict([(domain, {}) for domain in services])
# for j in tqdm(goals):
#     for g in goals:
#         for domain in services:
#             if g['goals'][domain] != {}:
#                 for slot in g['goals'][domain].keys():
#                     if slot not in ind_slot_cnt[domain]:
#                         ind_slot_cnt[domain][slot] = 1
#                     else:
#                         ind_slot_cnt[domain][slot] += 1
                        
# # self.ind_slot_dist = deepcopy(ind_slot_cnt)
# ind_slot_dist = deepcopy(ind_slot_cnt)
# for itk in ind_slot_cnt.keys():
#     for order in ind_slot_cnt[itk]:
#         # self.ind_slot_dist[order] = ind_slot_cnt[order] / sum(ind_slot_cnt.values())
#         ind_slot_dist[itk][order] = ind_slot_cnt[itk][order] / sum(ind_slot_cnt[itk].values())
    
