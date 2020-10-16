# -*- coding: utf-8 -*-
"""
@author: ryuichi takanobu
@modified: anubhav sachan
"""
import sys
import time
import logging
from utils import get_parser, init_logging_handler
from datamanager import DSTCDataManager, DataManager
from config import MultiWozConfig, DSTCSGDSConfig
from torch import multiprocessing as mp
from policy import Policy, DSTCPolicy
from learner import Learner, DSTCLearner
from controller import Controller, DSTCController
from agenda import UserAgenda
from rule import SystemRule

def worker_policy_sys(args, manager, config):
    init_logging_handler(args.log_dir, '_policy_sys')
    if args.config == 'multiwoz':
        print("MultiWoz Agent sys")
        agent = Policy(None, args, manager, config, 0, 'sys', True)
    elif args.config == 'dstcsgds':
        print("DSTC Agent sys")  
        agent = DSTCPolicy(None, args, manager, config, 0, 'sys', True)
    else:
        raise NotImplementedError('Policy sys of the dataset {} not implemented'.format(args.config))
    
    best = float('inf')
    for e in range(2):
        agent.imitating(e)
        best = agent.imit_test(e, best)

def worker_policy_usr(args, manager, config):
    init_logging_handler(args.log_dir, '_policy_usr')
    if args.config == 'multiwoz':
        print("MultiWoz Agent Usr")
        agent = Policy(None, args, manager, config, 0, 'usr', True)
    elif args.config == 'dstcsgds':
        print("DSTC Agent Usr")
        agent = DSTCPolicy(None, args, manager, config, 0, 'usr', True)
    else:
        raise NotImplementedError('Policy usr of the dataset {} not implemented'.format(args.config))
    
    best = float('inf')
    for e in range(2):
        agent.imitating(e)
        best = agent.imit_test(e, best)

def make_env(data_dir, config):
    controller = Controller(data_dir, config)
    return controller

def make_env_dstc(data_dir, config):
    controller = DSTCController(data_dir, config)
    return controller
    
def make_env_rule(data_dir, config):
    env = SystemRule(data_dir, config)
    return env

def make_env_agenda(data_dir, config):
    env = UserAgenda(data_dir, config)
    return env

if __name__ == '__main__':
    parser = get_parser()
    argv = sys.argv[1:]
    args, _ = parser.parse_known_args(argv)
    
    if args.config == 'multiwoz':
        print("MultiWoz Config")
        config = MultiWozConfig()
    elif args.config == 'dstcsgds':
        print("DSTC Config")
        config = DSTCSGDSConfig(args.data_dir)
    else:
        raise NotImplementedError('Config of the dataset {} not implemented'.format(args.config))

    init_logging_handler(args.log_dir)
    logging.debug(str(args))
    
    try:
        mp = mp.get_context('spawn')
    except RuntimeError:
        pass
    
    if args.pretrain:
        logging.debug('pretrain')
        if args.config == 'dstcsgds':
            print("DSTC Manager")
            manager = DSTCDataManager(args.data_dir, config)
        elif args.config == 'multiwoz':
            print("MultiWoz Manager")
            manager = DataManager(args.data_dir, config)
        processes = []
        process_args = (args, manager, config)
        processes.append(mp.Process(target=worker_policy_sys, args=process_args))
        processes.append(mp.Process(target=worker_policy_usr, args=process_args))
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        logging.debug('pre-training complete')
    elif args.test:
        logging.debug('test')
        logging.disable(logging.DEBUG)
    
        if args.config == 'multiwoz':
            agent = Learner(make_env, args, config, 1, infer = True)
        elif args.config == 'dstcsgds':
            agent = DSTCLearner(make_env_dstc, args, config, 1, infer = True)
            
        agent.load(args.load)
        agent.evaluate(args.test_case)
        
        # # test system policy with agenda
        # env = make_env_agenda(args.data_dir, config)
        # agent.evaluate_with_agenda(env, args.test_case)

        # # test user policy with rule
        # env = make_env_rule(args.data_dir, config)
        # agent.evaluate_with_rule(env, args.test_case)
                
    else: # training
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        logging.debug('train starts at {}'.format(current_time))
        if args.config == 'multiwoz':
            agent = Learner(make_env, args, config, args.process)
        elif args.config == 'dstcsgds':
            agent = DSTCLearner(make_env_dstc, args, config, args.process)
            
        best = agent.load(args.load)

        for i in range(args.epoch):
            print(i, 'update-start')
            agent.update(args.batchsz_traj, i)
            # validation
            print(i, 'UPDATE DONE')
            best = agent.update(args.batchsz, i, best)
            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.debug('epoch {} {}'.format(i, current_time))
