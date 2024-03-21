# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
import random

class Game():
    ''' 
    Create a game enviroment for Ferrero game.
    Game area is a 6 plus 8 np.array.
    
    '''
    def __init__(self,
                 ROW=8,
                 COL=8,
                 actions_list=['up', 'down', 'left', 'right'],
                 # actions_num=4*5*4,
                 actions_num=8*9*8,
                 episodes=1):
        ''' 
        Initialize game.

        Parameters
        ----------
        ROW (int) : Num of rows
        COL (int) : Num of column
        actions_list (str list): Consisting of 4 directions
        actions_num (int) : Num of actions (contain many ilegal actions)
        raw_action (dict): A raw action is represented by position and direction
        raw_action = {'pos': pos (tuple), 'direc': direc (int)}
        state (dict): A state stores observation and legal std_actions
        state = {'obs':obs, 'legal_actions':legal_actions}
        memory (list): A list of tuple (state, action, reward) 
        
        Returns
        -------
        None.

        '''
        self.ROW = ROW
        self.COL = COL
        self.actions_list = actions_list
        self.actions_num = actions_num
        self.episodes = episodes
        self.state = {'obs': None, 'legal_actions':None}
        self.memory = []
        
    def reset(self):
        ''' 
        Reset game state and episode memory
        '''
        ROW, COL = self.ROW, self.COL
        self.state['obs'] = np.ones((ROW, COL))
        initial_points = [(1, 0), (2, 0), (0, 1), (0, 2), (3, 0), (3, 1), (1, 3), (2, 3)]
        self.state['obs'][random.choice(initial_points)] = 0
        # self.state['obs'][np.random.randint(ROW), np.random.randint(COL)] = 0
        self.state['legal_actions'] = self.get_legal_actions(self.state)
        self.memory = []
    
    
    def set_state(self, state):
        self.state = state
    
    
    def get_state(self):
        return deepcopy(self.state)
    
        
    def is_end(self):
        '''
        Judge whether a game is end.

        Returns
        -------
        flag (bool)

        '''
        actions = self.get_legal_actions(self.state)
        return True if actions == [] else False

    def get_legal_actions(self, state) -> list:
        """
        Return all legal std_actions.

        Returns
        -------
        std_actions (list of int)

        """
        ROW, COL = self.ROW, self.COL 
        std_actions = []
        for i in range(ROW):
            for j in range(COL):
                if state['obs'][i, j] == 1:
                    if 0 <= i - 2 < ROW:
                        if state['obs'][i-1, j] == 1 and state['obs'][i-2, j] == 0: # up
                            std_actions.append( (i*COL+j)*4 + 0 )
                    if 0 <= i + 2 < ROW:
                        if state['obs'][i+1, j] == 1 and state['obs'][i+2, j] == 0: # down
                            std_actions.append( (i*COL+j)*4 + 1 )
                    if 0 <= j - 2 < COL:
                        if state['obs'][i, j-1] == 1 and state['obs'][i, j-2] == 0: # left
                            std_actions.append( (i*COL+j)*4 + 2 )
                    if 0 <= j + 2 < COL:
                        if state['obs'][i, j+1] == 1 and state['obs'][i, j+2] == 0: # right 
                            std_actions.append( (i*COL+j)*4 + 3 )
        return std_actions
    
    
    def get_legal_pos(self, pos):
        '''
        Return all legal positions given a selected position

        Parameters
        ----------
        pos (tuple) : (x, y)
            
        Returns
        -------
        legal_pos (list of tuple): [(x1, y1), (x2, y2),...] 

        '''
        ROW, COL = self.ROW, self.COL
        (x, y) = pos
        legal_pos = []
        if self.state['obs'][x, y] == 1:
            if 0 <= x - 2 < ROW:
                if self.state['obs'][x-1, y] == 1 and self.state['obs'][x-2, y] == 0: # up
                    legal_pos.append((x-2, y))
            if 0 <= x + 2 < ROW:
                if self.state['obs'][x+1, y] == 1 and self.state['obs'][x+2, y] == 0: # down
                    legal_pos.append((x+2, y))
            if 0 <= y - 2 < COL:
                if self.state['obs'][x, y-1] == 1 and self.state['obs'][x, y-2] == 0: # left
                    legal_pos.append((x, y-2))
            if 0 <= y + 2 < COL:
                if self.state['obs'][x, y+1] == 1 and self.state['obs'][x, y+2] == 0: # left
                    legal_pos.append((x, y+2))
        return legal_pos
                   
    
    def raw_to_std(self, raw_action):
        '''
        Change a raw action to a standard one.

        Parameters
        ----------
        raw_action (dict): {'pos': pos (tuple), 'direc': direc (int)}

        Returns
        -------
        std_action (int)

        '''
        COL = self.COL
        (x, y), direc = raw_action['pos'], raw_action['direc']
        return ( x*COL + y )*4 + direc
    
    
    def std_to_raw(self, std_action):
        '''
        Change a standard action to a raw one.

        Parameters
        ----------
        std_action (int)

        Returns
        -------
        raw_action (dict): {'pos': pos (tuple), 'direc': direc (int)}

        '''
        COL = self.COL
        direc = std_action % 4
        tmp = std_action // 4
        x, y = tmp // COL, tmp % COL
        return {'pos':(x, y), 'direc':direc}
        
        
    def step(self, std_action):
        '''
        Agent Interact with enviroment based on Morkov model.
        Agent takes an action, recieves reward, and changes envirooment to a new state.
        Autimatically store transition tuple (state, action, reward).
        If done, calculate q for each experience in episode memory.

        Parameters
        ----------
        std_action (int) 

        Returns
        -------
        state (dict) : {'obs':obs, 'legal_actions':legal_actions} 
        next_state (dict):  {'obs':obs, 'legal_actions':legal_actions}
        reward (int)
        done (bool)

        '''
        state = deepcopy(self.state) 
        raw_action = self.std_to_raw(std_action)
        (x, y), direc = raw_action['pos'], raw_action['direc']
        self.state['obs'][x, y] = 0
        if direc == 0: # up
            self.state['obs'][x-1, y] = 0
            self.state['obs'][x-2, y] = 1
        elif direc == 1: # down
            self.state['obs'][x+1, y] = 0
            self.state['obs'][x+2, y] = 1
        elif direc == 2: # left
            self.state['obs'][x, y-1] = 0
            self.state['obs'][x, y-2] = 1
        elif direc == 3: # right
            self.state['obs'][x, y+1] = 0
            self.state['obs'][x, y+2] = 1
        
        next_state = deepcopy(self.state)
        next_state['legal_actions'] = deepcopy(self.get_legal_actions(self.state))
        
        done = self.is_end()
        reward = 8 - self.state['obs'].sum() if done else 0
        self.memory.append((state, std_action, reward))
                    
        return state, std_action, next_state, reward, done
    
                
    def random_step(self):
        '''
        Randomly choice one legal action and interact with enviroment.

        '''
        legal_actions = self.state['legal_actions']
        action = random.choice(legal_actions)
        return self.step(action)
