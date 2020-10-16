# -*- coding: utf-8 -*-
"""
@author: ryuichi takanobu
@modified: anubhav sachan
"""

import os
import json
import logging
import torch
import torch.utils.data as data
import copy
from copy import deepcopy
from evaluator import MultiWozEvaluator, DSTCSGDSEvaluator
from utils import init_session, init_goal, state_vectorize, action_vectorize, \
    state_vectorize_user, action_vectorize_user, discard, reload
    
from utils import init_session_dstc, init_goal_dstc, state_vectorize_dstc, \
    state_vectorize_user_dstc

def expand_da(meta):
    for k, v in meta.items():
        domain, intent = k.split('-')
        if intent.lower() == "request":
            for pair in v:
                pair.insert(1, '?')
        else:
            counter = {}
            for pair in v:
                if pair[0] == 'none':
                    pair.insert(1, 'none')
                else:
                    if pair[0] in counter:
                        counter[pair[0]] += 1
                    else:
                        counter[pair[0]] = 1
                    pair.insert(1, str(counter[pair[0]]))


def add_domain_mask(data):
    f = open(('dstc-data/dstc_services.txt'), 'r')
    content = f.read()
    all_domains = content.split('\n')[:-1]
    f.close()
    # all_domains = ['Calendar', 'Hotels', 'RentalCars', 'Services', 'Trains', 'Music', 'Homes', 'Flights', 'Travel', 'Media', 'RideSharing', 'Alarm', 'Banks', 'Movies', 'Weather', 'Messaging', 'Payment', 'Buses', 'Restaurants', 'Events']
    parts = ["train", "valid", "test"]
    for part in parts:
        dataset = data[part]
        domains_in_order_dict = {} # {session_id:domains_in_order}
        domains_in_order = []
        current_session_id = ""
        for turn in dataset:
            session_id = turn["others"]["session_id"]
            if session_id != current_session_id:
                if current_session_id != "":
                    domains_in_order_dict[current_session_id] = domains_in_order
                domains_in_order = []
                current_session_id = session_id
            if "trg_user_action" in turn:
                user_das = turn["trg_user_action"]
                for user_da in user_das:
                    [domain, intent, slot] = user_da.split('~')
                    if domain in all_domains and domain not in domains_in_order:
                        domains_in_order.append(domain)
        domains_in_order_dict[current_session_id] = domains_in_order # for last dialog
        
        current_session_id = ""
        next_available_domain = ""
        invisible_domains = []
        for turn in dataset:
            session_id = turn["others"]["session_id"]
            if session_id != current_session_id:
                domains_in_order = domains_in_order_dict[session_id]
                if domains_in_order:
                    next_available_domain = domains_in_order[0]
                    invisible_domains = domains_in_order[1:]
                else:
                    next_available_domain = ""
                    invisible_domains = []
                current_session_id = session_id
            turn["next_available_domain"] = next_available_domain
            turn["invisible_domains"] = copy.copy(invisible_domains)

            if "trg_user_action" in turn:
                user_das = turn["trg_user_action"]
                for user_da in user_das:
                    [domain, intent, slot] = user_da.split('~')
                    if domain == next_available_domain:
                        if invisible_domains:
                            next_available_domain = invisible_domains[0]
                            invisible_domains.remove(next_available_domain)


class DataManager():
    """Offline data manager"""
    
    def __init__(self, data_dir, cfg):
        self.data = {}
        self.goal = {}
        
        self.data_dir_new = data_dir + '/processed_data'
        if os.path.exists(self.data_dir_new):
            logging.info('Load processed data file')
            for part in ['train','valid','test']:
                with open(self.data_dir_new + '/' + part + '.json', 'r') as f:
                    self.data[part] = json.load(f)
                with open(self.data_dir_new + '/' + part + '_goal.json', 'r') as f:
                    self.goal[part] = json.load(f)
        else:
            from dbquery import DBQuery
            db = DBQuery(data_dir, cfg)
            logging.info('Start preprocessing the dataset')
            self._build_data(data_dir, self.data_dir_new, cfg, db)

        for part in ['train', 'valid', 'test']:
            file_dir = self.data_dir_new + '/' + part + '_sys.pt'
            if not os.path.exists(file_dir):
                from dbquery import DBQuery
                db = DBQuery(data_dir, cfg)
                self.create_dataset_sys(part, file_dir, data_dir, cfg, db)
                
            file_dir = self.data_dir_new + '/' + part + '_usr.pt'
            if not os.path.exists(file_dir):
                from dbquery import DBQuery
                db = DBQuery(data_dir, cfg)
                self.create_dataset_usr(part, file_dir, data_dir, cfg, db)
                
            file_dir = self.data_dir_new + '/' + part + '_glo.pt'
            if not os.path.exists(file_dir):
                from dbquery import DBQuery
                db = DBQuery(data_dir, cfg)
                self.create_dataset_global(part, file_dir, data_dir, cfg, db)
            
    def _build_data(self, data_dir, data_dir_new, cfg, db):
        data_filename = data_dir + '/' + cfg.data_file
        with open(data_filename, 'r') as f:
            origin_data = json.load(f)
        
        for part in ['train','valid','test']:
            self.data[part] = []
            self.goal[part] = {}
            
        valList = []
        with open(data_dir + '/' + cfg.val_file) as f:
            for line in f:
                valList.append(line.split('.')[0])
        testList = []
        with open(data_dir + '/' + cfg.test_file) as f:
            for line in f:
                testList.append(line.split('.')[0])
            
        for k_sess in origin_data:
            sess = origin_data[k_sess]
            if k_sess in valList:
                part = 'valid'
            elif k_sess in testList:
                part = 'test'
            else:
                part = 'train'
            turn_data, session_data = init_session(k_sess, cfg)
            belief_state = turn_data['belief_state']
            goal_state = turn_data['goal_state']
            init_goal(session_data, goal_state, sess['goal'], cfg)
            self.goal[part][k_sess] = deepcopy(session_data)
            current_domain = ''
            book_domain = ''
            turn_data['trg_user_action'] = {}
            turn_data['trg_sys_action'] = {}
            
            for i, turn in enumerate(sess['log']):
                turn_data['others']['turn'] = i
                turn_data['others']['terminal'] = i + 2 >= len(sess['log'])
                da_origin = turn['dialog_act']
                expand_da(da_origin)
                turn_data['belief_state'] = deepcopy(belief_state) # from previous turn
                turn_data['goal_state'] = deepcopy(goal_state)

                if i % 2 == 0: # user
                    turn_data['sys_action'] = deepcopy(turn_data['trg_sys_action'])
                    del(turn_data['trg_sys_action'])
                    turn_data['trg_user_action'] = dict()
                    for domint in da_origin:
                        domain_intent = da_origin[domint]
                        _domint = domint.lower()
                        _domain, _intent = _domint.split('-')
                        if _domain in cfg.belief_domains:
                            current_domain = _domain
                        for slot, p, value in domain_intent:
                            _slot = slot.lower()
                            _value = value.strip()
                            _da = '-'.join((_domint, _slot))
                            if _da in cfg.da_usr:
                                turn_data['trg_user_action'][_da] = _value
                                if _intent == 'inform':
                                    inform_da = _domain+'-'+_slot
                                    if inform_da in cfg.inform_da:
                                        belief_state[_domain][_slot] = _value
                                    if inform_da in cfg.inform_da_usr and _slot in session_data[_domain] \
                                        and session_data[_domain][_slot] != '?':
                                        discard(goal_state[_domain], _slot)
                                elif _intent == 'request':
                                    request_da = _domain+'-'+_slot
                                    if request_da in cfg.request_da:
                                        belief_state[_domain][_slot] = '?'
                        
                else: # sys
                    book_status = turn['metadata']
                    for domain in cfg.belief_domains:
                        if book_status[domain]['book']['booked']:
                            entity = book_status[domain]['book']['booked'][0]
                            if 'booked' in belief_state[domain]:
                                continue
                            book_domain = domain
                            if domain in ['taxi', 'hospital', 'police']:
                                belief_state[domain]['booked'] = f'{domain}-booked'
                            elif domain == 'train':
                                found = db.query(domain, [('trainID', entity['trainID'])])
                                belief_state[domain]['booked'] = found[0]['ref']
                            else:
                                found = db.query(domain, [('name', entity['name'])])
                                belief_state[domain]['booked'] = found[0]['ref']
                    
                    turn_data['user_action'] = deepcopy(turn_data['trg_user_action'])
                    del(turn_data['trg_user_action'])
                    turn_data['others']['change'] = False
                    turn_data['trg_sys_action'] = dict()
                    for domint in da_origin:
                        domain_intent = da_origin[domint]
                        _domint = domint.lower()
                        _domain, _intent = _domint.split('-')
                        for slot, p, value in domain_intent:
                            _slot = slot.lower()
                            _value = value.strip()
                            _da = '-'.join((_domint, _slot, p))
                            if _da in cfg.da and current_domain:
                                if _slot == 'ref':
                                    turn_data['trg_sys_action'][_da] = belief_state[book_domain]['booked']
                                else:
                                    turn_data['trg_sys_action'][_da] = _value
                                if _intent in ['inform', 'recommend', 'offerbook', 'offerbooked', 'book']:
                                    inform_da = current_domain+'-'+_slot
                                    if inform_da in cfg.request_da:
                                        discard(belief_state[current_domain], _slot, '?')
                                    if inform_da in cfg.request_da_usr and _slot in session_data[current_domain] \
                                        and session_data[current_domain][_slot] == '?':
                                        goal_state[current_domain][_slot] = _value
                                elif _intent in ['nooffer', 'nobook']:
                                    # TODO: better transition
                                    for da in turn_data['user_action']:
                                        __domain, __intent, __slot = da.split('-')
                                        if __intent == 'inform' and __domain == current_domain:
                                            discard(belief_state[current_domain], __slot)
                                    turn_data['others']['change'] = True
                                    reload(goal_state, session_data, current_domain)
                
                if i + 1 == len(sess['log']):
                    turn_data['final_belief_state'] = belief_state
                    turn_data['final_goal_state'] = goal_state
                
                self.data[part].append(deepcopy(turn_data))

        add_domain_mask(self.data)
                                
        def _set_default(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError
        os.makedirs(data_dir_new)
        for part in ['train','valid','test']:
            with open(data_dir_new + '/' + part + '.json', 'w') as f:
                self.data[part] = json.dumps(self.data[part], default=_set_default)
                f.write(self.data[part])
                self.data[part] = json.loads(self.data[part])
            with open(data_dir_new + '/' + part + '_goal.json', 'w') as f:
                self.goal[part] = json.dumps(self.goal[part], default=_set_default)
                f.write(self.goal[part])
                self.goal[part] = json.loads(self.goal[part])
        
    def create_dataset_sys(self, part, file_dir, data_dir, cfg, db):
        datas = self.data[part]
        goals = self.goal[part]
        s, a, r, next_s, t = [], [], [], [], []
        evaluator = MultiWozEvaluator(data_dir)
        for idx, turn_data in enumerate(datas):
            if turn_data['others']['turn'] % 2 == 0:
                if turn_data['others']['turn'] == 0:
                    evaluator.add_goal(goals[turn_data['others']['session_id']])
                evaluator.add_usr_da(turn_data['trg_user_action'])
                continue
            if turn_data['others']['turn'] != 1:
                next_s.append(s[-1])
            
            s.append(torch.Tensor(state_vectorize(turn_data, cfg, db, True)))
            a.append(torch.Tensor(action_vectorize(turn_data['trg_sys_action'], cfg)))
            evaluator.add_sys_da(turn_data['trg_sys_action'])
            if turn_data['others']['terminal']:
                next_turn_data = deepcopy(turn_data)
                next_turn_data['others']['turn'] = -1
                next_turn_data['user_action'] = {}
                next_turn_data['sys_action'] = turn_data['trg_sys_action']
                next_turn_data['trg_sys_action'] = {}
                next_turn_data['belief_state'] = turn_data['final_belief_state']
                next_s.append(torch.Tensor(state_vectorize(next_turn_data, cfg, db, True)))
                reward = 20 if evaluator.task_success(False) else -5
                r.append(reward)
                t.append(1)
            else:
                reward = 0
                if evaluator.cur_domain:
                    for slot, value in turn_data['belief_state'][evaluator.cur_domain].items():
                        if value == '?':
                            for da in turn_data['trg_sys_action']:
                                d, i, k, p = da.split('-')
                                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and k == slot:
                                    break
                            else:
                                # not answer request
                                reward -= 1
                if not turn_data['trg_sys_action']:
                    reward -= 5
                r.append(reward)
                t.append(0)
                
        torch.save((s, a, r, next_s, t), file_dir)
        
    def create_dataset_usr(self, part, file_dir, data_dir, cfg, db):
        datas = self.data[part]
        goals = self.goal[part]
        s, a, r, next_s, t = [], [], [], [], []
        evaluator = MultiWozEvaluator(data_dir)
        current_goal = None
        for idx, turn_data in enumerate(datas):
            if turn_data['others']['turn'] % 2 == 1:
                evaluator.add_sys_da(turn_data['trg_sys_action'])
                continue
            
            if turn_data['others']['turn'] == 0:
                current_goal = goals[turn_data['others']['session_id']]
                evaluator.add_goal(current_goal)
            else:
                next_s.append(s[-1])
            if turn_data['others']['change'] and evaluator.cur_domain:
                if 'final' in current_goal[evaluator.cur_domain]:
                    for key in current_goal[evaluator.cur_domain]['final']:
                        current_goal[evaluator.cur_domain][key] = current_goal[evaluator.cur_domain]['final'][key]
                    del(current_goal[evaluator.cur_domain]['final'])
            turn_data['user_goal'] = deepcopy(current_goal)
            
            s.append(torch.Tensor(state_vectorize_user(turn_data, cfg, evaluator.cur_domain)))
            a.append(torch.Tensor(action_vectorize_user(turn_data['trg_user_action'], turn_data['others']['terminal'], cfg)))
            evaluator.add_usr_da(turn_data['trg_user_action'])
            if turn_data['others']['terminal']:
                next_turn_data = deepcopy(turn_data)
                next_turn_data['others']['turn'] = -1
                next_turn_data['user_action'] = turn_data['trg_user_action']
                next_turn_data['sys_action'] = datas[idx+1]['trg_sys_action']
                next_turn_data['trg_user_action'] = {}
                next_turn_data['goal_state'] = datas[idx+1]['final_goal_state']
                next_s.append(torch.Tensor(state_vectorize_user(next_turn_data, cfg, evaluator.cur_domain)))
                reward = 20 if evaluator.inform_F1(ansbysys=False)[1] == 1. else -5
                r.append(reward)
                t.append(1)
            else:
                reward = 0
                if evaluator.cur_domain:
                    for da in turn_data['trg_user_action']:
                        d, i, k = da.split('-')
                        if i == 'request':
                            for slot, value in turn_data['goal_state'][d].items():
                                if value != '?' and slot in turn_data['user_goal'][d]\
                                    and turn_data['user_goal'][d][slot] != '?':
                                    # request before express constraint
                                    reward -= 1
                if not turn_data['trg_user_action']:
                    reward -= 5
                r.append(reward)
                t.append(0)
                
        torch.save((s, a, r, next_s, t), file_dir)
        
    def create_dataset_global(self, part, file_dir, data_dir, cfg, db):
        datas = self.data[part]
        goals = self.goal[part]
        s_usr, s_sys, r_g, next_s_usr, next_s_sys, t = [], [], [], [], [], []
        evaluator = MultiWozEvaluator(data_dir)
        for idx, turn_data in enumerate(datas):
            if turn_data['others']['turn'] % 2 == 0:
                if turn_data['others']['turn'] == 0:
                    current_goal = goals[turn_data['others']['session_id']]
                    evaluator.add_goal(current_goal)
                else:
                    next_s_usr.append(s_usr[-1])
                
                if turn_data['others']['change'] and evaluator.cur_domain:
                    if 'final' in current_goal[evaluator.cur_domain]:
                        for key in current_goal[evaluator.cur_domain]['final']:
                            current_goal[evaluator.cur_domain][key] = current_goal[evaluator.cur_domain]['final'][key]
                        del(current_goal[evaluator.cur_domain]['final'])
                
                turn_data['user_goal'] = deepcopy(current_goal)
                s_usr.append(torch.Tensor(state_vectorize_user(turn_data, cfg, evaluator.cur_domain)))
                evaluator.add_usr_da(turn_data['trg_user_action'])
                    
                if turn_data['others']['terminal']:
                    next_turn_data = deepcopy(turn_data)
                    next_turn_data['others']['turn'] = -1
                    next_turn_data['user_action'] = turn_data['trg_user_action']
                    next_turn_data['sys_action'] = datas[idx+1]['trg_sys_action']
                    next_turn_data['trg_user_action'] = {}
                    next_turn_data['goal_state'] = datas[idx+1]['final_goal_state']
                    next_s_usr.append(torch.Tensor(state_vectorize_user(next_turn_data, cfg, evaluator.cur_domain)))
            
            else:
                if turn_data['others']['turn'] != 1:
                    next_s_sys.append(s_sys[-1])

                s_sys.append(torch.Tensor(state_vectorize(turn_data, cfg, db, True)))
                evaluator.add_sys_da(turn_data['trg_sys_action'])
            
                if turn_data['others']['terminal']:
                    next_turn_data = deepcopy(turn_data)
                    next_turn_data['others']['turn'] = -1
                    next_turn_data['user_action'] = {}
                    next_turn_data['sys_action'] = turn_data['trg_sys_action']
                    next_turn_data['trg_sys_action'] = {}
                    next_turn_data['belief_state'] = turn_data['final_belief_state']
                    next_s_sys.append(torch.Tensor(state_vectorize(next_turn_data, cfg, db, True)))
                    reward_g = 20 if evaluator.task_success() else -5
                    r_g.append(reward_g)
                    t.append(1)
                else:
                    reward_g = 5 if evaluator.cur_domain and evaluator.domain_success(evaluator.cur_domain) else -1
                    r_g.append(reward_g)
                    t.append(0)
                
        torch.save((s_usr, s_sys, r_g, next_s_usr, next_s_sys, t), file_dir)
        
    def create_dataset_policy(self, part, batchsz, cfg, db, character='sys'):
        assert part in ['train', 'valid', 'test']
        logging.debug('start loading {}'.format(part))
        
        if character == 'sys':
            file_dir = self.data_dir_new + '/' + part + '_sys.pt'
        elif character == 'usr':
            file_dir = self.data_dir_new + '/' + part + '_usr.pt'     
        else:
            raise NotImplementedError('Unknown character {}'.format(character))
        
        s, a, *_ = torch.load(file_dir)
        new_s, new_a = [], []
        for state, action in zip(s, a):
            if action.nonzero().size(0):
                new_s.append(state)
                new_a.append(action)
        dataset = Dataset_Policy(new_s, new_a)
        dataloader = data.DataLoader(dataset, batchsz, True)
        
        logging.debug('finish loading {}'.format(part))
        return dataloader
            
    def create_dataset_vnet(self, part, batchsz, cfg, db):
        assert part in ['train', 'valid', 'test']
        logging.debug('start loading {}'.format(part))
        
        file_dir_1 = self.data_dir_new + '/' + part + '_sys.pt'
        file_dir_2 = self.data_dir_new + '/' + part + '_usr.pt'
        file_dir_3 = self.data_dir_new + '/' + part + '_glo.pt'
        
        s, _, r, next_s, t = torch.load(file_dir_1)
        dataset_sys = Dataset_Vnet(s, r, next_s, t)
        dataloader_sys = data.DataLoader(dataset_sys, batchsz, True)
        
        s, _, r, next_s, t = torch.load(file_dir_2)
        dataset_usr = Dataset_Vnet(s, r, next_s, t)
        dataloader_usr = data.DataLoader(dataset_usr, batchsz, True)
        
        s_usr, s_sys, r_g, next_s_usr, next_s_sys, t = torch.load(file_dir_3)
        dataset_global = Dataset_Vnet_G(s_usr, s_sys, r_g, next_s_usr, next_s_sys, t)
        dataloader_global = data.DataLoader(dataset_global, batchsz, True)
        
        logging.debug('finish loading {}'.format(part))
        return dataloader_sys, dataloader_usr, dataloader_global

class DSTCDataManager():
    """Offline data manager for DSTC"""
    
    def __init__(self, data_dir, cfg):
        self.data = {}
        self.goal = {}
        
        self.data_dir_new = data_dir + '/processed_data'
        if os.path.exists(self.data_dir_new):
            logging.info('Load processed data file')
            for part in ['train','valid','test']:
                with open(self.data_dir_new + '/' + part + '.json', 'r') as f:
                    self.data[part] = json.load(f)
                with open(self.data_dir_new + '/' + part + '_goal.json', 'r') as f:
                    self.goal[part] = json.load(f)
        else:
            logging.info('Start preprocessing the dataset')
            self._build_data(data_dir, self.data_dir_new, cfg)

        for part in ['train', 'valid', 'test']:
            file_dir = self.data_dir_new + '/' + part + '_sys.pt'
            if not os.path.exists(file_dir):
                logging.info('Creating ' + str(part) + ' dataset_sys')
                self.create_dataset_sys(part, file_dir, data_dir, cfg)
                
            file_dir = self.data_dir_new + '/' + part + '_usr.pt'
            if not os.path.exists(file_dir):
                logging.info('Creating ' + str(part) + ' dataset_usr')
                self.create_dataset_usr(part, file_dir, data_dir, cfg)
                
            file_dir = self.data_dir_new + '/' + part + '_glo.pt'
            if not os.path.exists(file_dir):
                logging.info('Creating ' + str(part) + ' dataset_glo')
                self.create_dataset_global(part, file_dir, data_dir, cfg)
            
            
    def _build_data(self, data_dir, data_dir_new, cfg):
        # data_filename = data_dir + '/' + cfg.data_file
        train_filename = data_dir + '/' + cfg.train_file
        valid_filename = data_dir + '/' + cfg.valid_file
        test_filename = data_dir + '/' + cfg.test_file

        train_goals_filename = data_dir + '/' + cfg.train_goals_file
        valid_goals_filename = data_dir + '/' + cfg.valid_goals_file
        test_goals_filename = data_dir + '/' + cfg.test_goals_file  
        
        train_schema_filename = 'dstc8-schema-guided-dialogue-master/train/schema.json'
        valid_schema_filename = 'dstc8-schema-guided-dialogue-master/dev/schema.json'
        test_schema_filename = 'dstc8-schema-guided-dialogue-master/test/schema.json'
        
        for part in ['train','valid','test']:
            self.data[part] = []
            self.goal[part] = {}
            
            if part == 'train':
                with open(train_filename, 'r') as f:
                    origin_data = json.load(f)
                with open(train_goals_filename, 'r') as fg:
                    goals_data = json.load(fg)
                with open(train_schema_filename, 'r') as fs:
                    schema_data = json.load(fs)
                    
            elif part == 'valid':
                with open(valid_filename, 'r') as f:
                    origin_data = json.load(f)
                with open(valid_goals_filename, 'r') as fg:
                    goals_data = json.load(fg)
                with open(valid_schema_filename, 'r') as fs:
                    schema_data = json.load(fs)
                    
            elif part == 'test':
                with open(test_filename, 'r') as f:
                    origin_data = json.load(f)
                with open(test_goals_filename, 'r') as fg:
                    goals_data = json.load(fg)
                with open(test_schema_filename, 'r') as fs:
                    schema_data = json.load(fs)
                    
            for conversation in origin_data:
                k_sess = conversation['dialogue_id']
                sess = conversation['turns']
                    
                print(str(part) + ' ' + str(k_sess))
                    
                turn_data, session_data = init_session_dstc(k_sess, cfg)
                belief_state = turn_data['belief_state']
                goal_state = turn_data['goal_state']
                
                turn_data['others']['services'] = []
                
                for fd in conversation['services']:
                    turn_data['others']['services'].append(fd)
                
                for gconv in goals_data:
                    if gconv['dialogue_id'] == conversation['dialogue_id']:
                        gdict = gconv['goals']
                            
                init_goal_dstc(session_data, goal_state, gdict, cfg) 
                self.goal[part][k_sess] = deepcopy(session_data)
    
                turn_data['trg_user_action'] = {}
                turn_data['trg_sys_action'] = {}
                
                turn_data['belief_state'] = deepcopy(belief_state) # from previous turn
                turn_data['goal_state'] = deepcopy(goal_state)  
                
                for i, turn in enumerate(sess):
                    turn_data['others']['turn'] = i
                    turn_data['others']['terminal'] = i + 2 >= len(sess)
                        
                    da_origin = []
                    for fr in turn['frames']:
                        serve_dia = fr['service']
                        da_origin = [serve_dia]
                        for item in fr['actions']:
                            da_origin.append(item) 

                        
                    if i % 2 == 0: # user
                        turn_data['sys_action'] = deepcopy(turn_data['trg_sys_action'])
                        del(turn_data['trg_sys_action'])
                        turn_data['trg_user_action'] = dict()
                        for c, item in enumerate(da_origin):
                            if c > 0:
                                key = '~'.join((str(da_origin[0]).lower(), str(da_origin[c]['act']).lower(), str(da_origin[c]['slot']).lower()))
                                for it in da_origin[c]['values']:
                                    turn_data['trg_user_action'][key] = it 
                                    
                        # Updation of belief_state
                        for fr in turn['frames']:
                            lsv = list(fr['state']['slot_values'].keys())
                            for item in lsv:
                                for it in fr['state']['slot_values'][item]:
                                    turn_data['belief_state'][da_origin[0]][item] = it
                            
                            if i == 0:  # initialization of turn_data['goal_state'][da_origin[0]]
                                for itg in schema_data:
                                    if itg['service_name'] == da_origin[0]:
                                        for inte in itg['intents']:
                                            if inte['name'] == fr['state']['active_intent']:
                                                for reqd in inte['required_slots']:
                                                    turn_data['goal_state'][da_origin[0]][reqd] = '?'
                                                    
                            else:   #updation of goal state with turn
                                gsa = list(turn_data['goal_state'][da_origin[0]].keys())
                                bsa = list(turn_data['belief_state'][da_origin[0]].keys())
                                for iteg in gsa:
                                    for iteb in bsa:
                                        if iteg == iteb:
                                            turn_data['goal_state'][da_origin[0]][iteg] = turn_data['belief_state'][da_origin[0]][iteg]
                                
                                for itegp in gsa:
                                    self.goal[part][k_sess][da_origin[0]][itegp] = turn_data['goal_state'][da_origin[0]][itegp]
                                
                                
                    elif i % 2 != 0: # system
                        turn_data['user_action'] = deepcopy(turn_data['trg_user_action'])
                        del(turn_data['trg_user_action'])
                        turn_data['others']['change'] = False
                        turn_data['trg_sys_action'] = dict()
                            
                        for c, item in enumerate(da_origin):
                            if c > 0:
                                if da_origin[c]['slot'] == '':
                                    da_origin[c]['slot'] = None
                                    
                                key = '~'.join((str(da_origin[0]).lower(), str(da_origin[c]['act']).lower(), str(da_origin[c]['slot']).lower()))
                                if da_origin[c]['values'] == []:
                                    turn_data['trg_sys_action'][key] = '_blank'
                                else:
                                    for it in da_origin[c]['values']:
                                        turn_data['trg_sys_action'][key] = it               
                                
                    if i + 1 == len(sess):
                        turn_data['final_belief_state'] = turn_data['belief_state']
                        turn_data['final_goal_state'] = turn_data['goal_state']
                    
                    self.data[part].append(deepcopy(turn_data))
                    
        add_domain_mask(self.data)
                                
        def _set_default(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError
            
        os.makedirs(data_dir_new)
        for part in ['train','valid','test']:
            print('saving ' + str(part) + '.json')
            with open(data_dir_new + '/' + part + '.json', 'w') as f:
                self.data[part] = json.dumps(self.data[part], indent = 2)
                f.write(self.data[part])
                self.data[part] = json.loads(self.data[part])
            with open(data_dir_new + '/' + part + '_goal.json', 'w') as f:
                self.goal[part] = json.dumps(self.goal[part], indent = 2)
                f.write(self.goal[part])
                self.goal[part] = json.loads(self.goal[part])
        
    def create_dataset_sys(self, part, file_dir, data_dir, cfg):
        
        datas = self.data[part]
        goals = self.goal[part]
        s, a, r, next_s, t = [], [], [], [], []
        
        evaluator = DSTCSGDSEvaluator(data_dir)
        
        for idx, turn_data in enumerate(datas):
            if turn_data['others']['turn'] % 2 == 0:
                if turn_data['others']['turn'] == 0:
                    evaluator.add_goal(goals[turn_data['others']['session_id']])
                evaluator.add_usr_da(turn_data['trg_user_action'])
                continue
            if turn_data['others']['turn'] != 1:
                next_s.append(s[-1])
            s.append(torch.Tensor(state_vectorize_dstc(turn_data, cfg)))
            a.append(torch.Tensor(action_vectorize(turn_data['trg_sys_action'], cfg)))
            evaluator.add_sys_da(turn_data['trg_sys_action'])
            if turn_data['others']['terminal'] == True:
                next_turn_data = deepcopy(turn_data)
                next_turn_data['others']['turn'] = -1
                next_turn_data['user_action'] = {}
                next_turn_data['sys_action'] = turn_data['trg_sys_action']
                next_turn_data['trg_sys_action'] = {}
                next_turn_data['belief_state'] = turn_data['final_belief_state']
                next_s.append(torch.Tensor(state_vectorize_dstc(next_turn_data, cfg)))
                #define task_success as, final_goal_state's keys has all the values
                reward = 20 if evaluator.task_success(turn_data) else -5
                r.append(reward)
                t.append(1)
            elif turn_data['others']['terminal'] == False:
                reward = 0
                for curr_dom in turn_data['others']['services']:
                    for slot, value in turn_data['goal_state'][curr_dom].items():
                        if value == '?':
                            for da in turn_data['trg_sys_action']:
                                d, i, k = da.split('~')
                                ls = ['inform', 'request', 'notify_success', 'offer', 'offer_intent']
                                if i in ls and k == slot:
                                    break
                                else:
                                    # not answer request
                                    reward -= 1

                if not turn_data['trg_sys_action']:
                    reward -= 5
                r.append(reward)
                t.append(0)     
        print('Saving ' + str(part) + ' dataset_sys')
        torch.save((s, a, r, next_s, t), file_dir)
        
    def create_dataset_usr(self, part, file_dir, data_dir, cfg):
        datas = self.data[part]
        goals = self.goal[part]
        s, a, r, next_s, t = [], [], [], [], []
        
        evaluator = DSTCSGDSEvaluator(data_dir)
        
        current_goal = None
        for idx, turn_data in enumerate(datas):
            if turn_data['others']['turn'] % 2 == 1:
                evaluator.add_sys_da(turn_data['trg_sys_action'])
                continue
            
            if turn_data['others']['turn'] == 0:
                current_goal = goals[turn_data['others']['session_id']]
                evaluator.add_goal(current_goal)
            else:
                next_s.append(s[-1])
            # if turn_data['others']['change'] and evaluator.cur_domain:
            #     if 'final' in current_goal[evaluator.cur_domain]:
            #         for key in current_goal[evaluator.cur_domain]['final']:
            #             current_goal[evaluator.cur_domain][key] = current_goal[evaluator.cur_domain]['final'][key]
            #         del(current_goal[evaluator.cur_domain]['final'])
            turn_data['user_goal'] = deepcopy(current_goal)
            
            s.append(torch.Tensor(state_vectorize_user_dstc(turn_data, cfg)))
            a.append(torch.Tensor(action_vectorize_user(turn_data['trg_user_action'], turn_data['others']['terminal'], cfg)))
            evaluator.add_usr_da(turn_data['trg_user_action'])
            if turn_data['others']['terminal'] == True:
                next_turn_data = deepcopy(turn_data)
                next_turn_data['others']['turn'] = -1
                next_turn_data['user_action'] = turn_data['trg_user_action']
                next_turn_data['sys_action'] = datas[idx+1]['trg_sys_action']
                next_turn_data['trg_user_action'] = {}
                next_turn_data['goal_state'] = datas[idx+1]['final_goal_state']
                next_s.append(torch.Tensor(state_vectorize_user_dstc(next_turn_data, cfg)))
                reward = 20 if evaluator.inform_F1_dstc(ansbysys=False)[1] == 1. else -5
                r.append(reward)
                t.append(1)
            elif turn_data['others']['terminal'] == False:
                reward = 0
                for curr_dom in turn_data['others']['services']:
                    for da in turn_data['trg_user_action']:
                        d, i, k = da.split('~')
                        if i == 'request':
                            for slot, value in turn_data['goal_state'][curr_dom].items():
                                if value != '?' and slot in turn_data['user_goal'][curr_dom]\
                                    and turn_data['user_goal'][curr_dom][slot] != '?':
                                    # request before express constraint
                                    reward -= 1
                if not turn_data['trg_user_action']:
                    reward -= 5
                r.append(reward)
                t.append(0)
        print('Saving ' + str(part) + ' dataset_user')        
        torch.save((s, a, r, next_s, t), file_dir)
        
    def create_dataset_global(self, part, file_dir, data_dir, cfg):
        datas = self.data[part]
        goals = self.goal[part]
        s_usr, s_sys, r_g, next_s_usr, next_s_sys, t = [], [], [], [], [], []
        evaluator = DSTCSGDSEvaluator(data_dir)
        for idx, turn_data in enumerate(datas):
            if turn_data['others']['turn'] % 2 == 0:
                if turn_data['others']['turn'] == 0:
                    current_goal = goals[turn_data['others']['session_id']]
                    evaluator.add_goal(current_goal)
                else:
                    next_s_usr.append(s_usr[-1])
                
                # if turn_data['others']['change'] and evaluator.cur_domain:
                #     if 'final' in current_goal[evaluator.cur_domain]:
                #         for key in current_goal[evaluator.cur_domain]['final']:
                #             current_goal[evaluator.cur_domain][key] = current_goal[evaluator.cur_domain]['final'][key]
                #         del(current_goal[evaluator.cur_domain]['final'])
                
                turn_data['user_goal'] = deepcopy(current_goal)
                s_usr.append(torch.Tensor(state_vectorize_user_dstc(turn_data, cfg)))
                evaluator.add_usr_da(turn_data['trg_user_action'])
                    
                if turn_data['others']['terminal']:
                    next_turn_data = deepcopy(turn_data)
                    next_turn_data['others']['turn'] = -1
                    next_turn_data['user_action'] = turn_data['trg_user_action']
                    next_turn_data['sys_action'] = datas[idx+1]['trg_sys_action']
                    next_turn_data['trg_user_action'] = {}
                    next_turn_data['goal_state'] = datas[idx+1]['final_goal_state']
                    next_s_usr.append(torch.Tensor(state_vectorize_user_dstc(next_turn_data, cfg)))
            
            else:
                if turn_data['others']['turn'] != 1:
                    next_s_sys.append(s_sys[-1])

                s_sys.append(torch.Tensor(state_vectorize_dstc(turn_data, cfg)))
                evaluator.add_sys_da(turn_data['trg_sys_action'])
            
                if turn_data['others']['terminal']:
                    next_turn_data = deepcopy(turn_data)
                    next_turn_data['others']['turn'] = -1
                    next_turn_data['user_action'] = {}
                    next_turn_data['sys_action'] = turn_data['trg_sys_action']
                    next_turn_data['trg_sys_action'] = {}
                    next_turn_data['belief_state'] = turn_data['final_belief_state']
                    next_s_sys.append(torch.Tensor(state_vectorize_dstc(next_turn_data, cfg)))
                    reward_g = 20 if evaluator.task_success(turn_data) else -5
                    r_g.append(reward_g)
                    t.append(1)
                else:
                    reward_g = 5 if evaluator.domain_success_dstc(turn_data) else -1
                    r_g.append(reward_g)
                    t.append(0)
        print(r_g)  
        print('Saving ' + str(part) + ' dataset_glo') 
        torch.save((s_usr, s_sys, r_g, next_s_usr, next_s_sys, t), file_dir)
        
    def create_dataset_policy(self, part, batchsz, cfg, character='sys'):
        assert part in ['train', 'valid', 'test']
        logging.debug('start loading {}'.format(part))
        
        if character == 'sys':
            file_dir = self.data_dir_new + '/' + part + '_sys.pt'
        elif character == 'usr':
            file_dir = self.data_dir_new + '/' + part + '_usr.pt'     
        else:
            raise NotImplementedError('Unknown character {}'.format(character))
        
        s, a, *_ = torch.load(file_dir)
        new_s, new_a = [], []
        for state, action in zip(s, a):
            if action.nonzero().size(0):
                new_s.append(state)
                new_a.append(action)
        dataset = Dataset_Policy(new_s, new_a)
        dataloader = data.DataLoader(dataset, batchsz, True)
        
        logging.debug('finish loading {}'.format(part))
        return dataloader
            
    def create_dataset_vnet(self, part, batchsz, cfg, db):
        assert part in ['train', 'valid', 'test']
        logging.debug('start loading {}'.format(part))
        
        file_dir_1 = self.data_dir_new + '/' + part + '_sys.pt'
        file_dir_2 = self.data_dir_new + '/' + part + '_usr.pt'
        file_dir_3 = self.data_dir_new + '/' + part + '_glo.pt'
        
        s, _, r, next_s, t = torch.load(file_dir_1)
        dataset_sys = Dataset_Vnet(s, r, next_s, t)
        dataloader_sys = data.DataLoader(dataset_sys, batchsz, True)
        
        s, _, r, next_s, t = torch.load(file_dir_2)
        dataset_usr = Dataset_Vnet(s, r, next_s, t)
        dataloader_usr = data.DataLoader(dataset_usr, batchsz, True)
        
        s_usr, s_sys, r_g, next_s_usr, next_s_sys, t = torch.load(file_dir_3)
        dataset_global = Dataset_Vnet_G(s_usr, s_sys, r_g, next_s_usr, next_s_sys, t)
        dataloader_global = data.DataLoader(dataset_global, batchsz, True)
        
        logging.debug('finish loading {}'.format(part))
        return dataloader_sys, dataloader_usr, dataloader_global

       

class Dataset_Policy(data.Dataset):
    def __init__(self, s, a):
        self.s = s
        self.a = a
        self.num_total = len(s)
    
    def __getitem__(self, index):
        s = self.s[index]
        a = self.a[index]
        return s, a
    
    def __len__(self):
        return self.num_total

class Dataset_Vnet(data.Dataset):
    def __init__(self, s, r, next_s, t):
        self.s = s
        self.r = r
        self.next_s = next_s
        self.t = t
        self.num_total = len(s)
    
    def __getitem__(self, index):
        s = self.s[index]
        r = self.r[index]
        next_s = self.next_s[index]
        t = self.t[index]
        return s, r, next_s, t
    
    def __len__(self):
        return self.num_total 

class Dataset_Vnet_G(data.Dataset):
    def __init__(self, s_usr, s_sys, r, next_s_usr, next_s_sys, t):
        self.s_usr = s_usr
        self.s_sys = s_sys
        self.r = r
        self.next_s_usr = next_s_usr
        self.next_s_sys = next_s_sys
        self.t = t
        self.num_total = len(s_sys)
    
    def __getitem__(self, index):
        s_usr = self.s_usr[index]
        s_sys = self.s_sys[index]
        r = self.r[index]
        next_s_usr = self.next_s_usr[index]
        next_s_sys = self.next_s_sys[index]
        t = self.t[index]
        return s_usr, s_sys, r, next_s_usr, next_s_sys, t
    
    def __len__(self):
        return self.num_total
