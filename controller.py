# -*- coding: utf-8 -*-
"""
@author: ryuichi takanobu
@modified: anubhav sachan
"""

from utils import init_goal, init_session, init_session_dstc, init_goal_dstc
from tracker import StateTracker, DSTCStateTracker
from goal_generator import GoalGenerator, DSTCGoalGenerator

class Controller(StateTracker):
    def __init__(self, data_dir, config):
        super(Controller, self).__init__(data_dir, config)
        self.goal_gen = GoalGenerator(data_dir, config,
                                      goal_model_path='processed_data/goal_model.pkl',
                                      corpus_path=config.data_file) # data_file needs to have train, test and dev parts
            
    def reset(self, random_seed=None):
        """
        init a user goal and return init state
        """
        self.time_step = 0
        self.topic = ''
        self.goal = self.goal_gen.get_user_goal(random_seed)
        
        dummy_state, dummy_goal = init_session(-1, self.cfg)
        init_goal(dummy_goal, dummy_state['goal_state'], self.goal, self.cfg)

        domain_ordering = self.goal['domain_ordering']
        dummy_state['next_available_domain'] = domain_ordering[0]
        dummy_state['invisible_domains'] = domain_ordering[1:]
    
        dummy_state['user_goal'] = dummy_goal
        self.evaluator.add_goal(dummy_goal)
        
        return dummy_state
        
    def step_sys(self, s, sys_a):
        """
        interact with simulator for one sys-user turn
        """
        # update state with sys_act
        current_s = self.update_belief_sys(s, sys_a)
        
        return current_s
    
    def step_usr(self, s, usr_a):
        current_s = self.update_belief_usr(s, usr_a)
        terminal = current_s['others']['terminal']
        return current_s, terminal
    
    
class DSTCController(DSTCStateTracker):
    def __init__(self, data_dir, config):
        super(DSTCController, self).__init__(data_dir, config)
        self.goal_gen = DSTCGoalGenerator(data_dir, config,
                                          goal_model_path='processed_data/goal_model_dstc.pkl') # data_file needs to have train, test and dev parts
        # self.evaluator = DSTCSGDSEvaluator(data_dir)    
        
    def reset(self, random_seed=None):
        """
        init a user goal and return init state
        """
        self.time_step = 0
        self.topic = ''
        self.goal = self.goal_gen.get_user_goal(random_seed)
        
        dummy_state, dummy_goal = init_session_dstc(-1, self.cfg)
        init_goal_dstc(dummy_goal, dummy_state['goal_state'], self.goal, self.cfg)

        domain_ordering = self.goal['domain_ordering']
        dummy_state['next_available_domain'] = domain_ordering[0]
        dummy_state['invisible_domains'] = domain_ordering[1:]
    
        dummy_state['user_goal'] = dummy_goal
        self.evaluator.add_goal_dstc(self.goal)
        
        return dummy_state
        
    def step_sys(self, s, sys_a):
        """
        interact with simulator for one sys-user turn
        """
        # update state with sys_act
        current_s = self.update_belief_sys(s, sys_a)
        
        return current_s
    
    def step_usr(self, s, usr_a):
        current_s = self.update_belief_usr(s, usr_a)
        terminal = current_s['others']['terminal']
        return current_s, terminal