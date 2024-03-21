# -*- coding: utf-8 -*-
import numpy as np
import pygame # render the game state
import time # control AI step internal
import csv
import sys

from game import Game
from mcts_pure import Mcts
# from mcts_wu_uct import Mcts


class UserInterface:
    '''
    Use pygame to render game enviroment.
    In general, a game object Game() is created in initialization of the UserInterface object.
    Function run() provide main loop for refreshing the screen,
    which contains 3 basic function: processInput(), update(), and render().
    processInput() : monitor device input in order to store control orders or quit the game.
    update() : execute stored control orders and update game state.
    render() : render game state.
    
    '''
    def __init__(self):
        pygame.init()
        
        # Create Game object
        self.game = Game(episodes=10)
        self.game.reset()
        self.ROW, self.COL = self.game.ROW, self.game.COL
        
        # Create Agent object
        self.agent = Mcts()
        self.agent.set_env_model(Game())
        
        # Record scores
        self.scores = []
        
        # General args
        self.BoardColor = (246, 234, 219) # board color
        self.BgColor = (246, 234, 219) # Background color
        self.LineColor = ((224, 172, 105)) # Line Color
        self.INTERVAL = 0 # AI moving interval
        self.FPS = 1 # frame per second
        self.SIZE = 75 # basic size
        
        # Basic configuration
        # self.window = pygame.display.set_mode((self.COL * self.SIZE, self.ROW * self.SIZE)) # Set window size
        # self.window = pygame.display.set_mode((670, 300)) # Set window size
        self.window = pygame.display.set_mode((1340, 600)) # Set window size
        pygame.display.set_caption('剩下最少🍓') # Set window name
        self.running = True # UI running flag
        self.clock = pygame.time.Clock() # FPS control
        self.time = time.time() # record time
        
        # Switch mode
        # AI will move only if human_mode == False and AI_mode == True
        self.human_mode = False  # whether allow human action
        self.AI_mode = True    # whether allow AI action
        
        # Store human input
        # if select['action'] is not none, it will be executed in update()
        self.select = {'pos':None, 'legal_pos':[], 'action':None}
        
        # Load image
        self.img = pygame.image.load('texture/strawberry.png').convert_alpha()
        self.img = pygame.transform.smoothscale(self.img, (0.8*self.SIZE, 0.8*self.SIZE))

        self.img_legal = pygame.image.load('texture/strawberry.png').convert_alpha()

        # Load font
        self.font = pygame.font.Font('texture/BD_Cartoon_Shout.ttf', 17)
        
        # Create text
        self.select_text = self.font.render('select', True, (220,20,60))
    
    
    def record_score(self, score):
        self.scores.append(score)
        
    
    def save_scores_as_csv(self):
        with open('results/mcts_scores/scores.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.scores)
        
        
    def processInput(self):
        for event in pygame.event.get():
            # Press cancel button on the top right corner to quit the game
            if event.type == pygame.QUIT:
                self.running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN and self.human_mode: # Press mouse
                x, y = pygame.mouse.get_pos()            
                if event.button == 1: # Press left mouse button to select 
                    print('left click: (%d,%d)' % (x, y))
                    row = y // self.SIZE
                    col = x // self.SIZE
                    
                    if (row, col) in self.select['legal_pos']:
                        x, y = self.select['pos']
                        a = None
                        if row == x - 2 and col == y:
                            a = {'pos':(x,y), 'direc':0}
                        elif row == x + 2 and col == y:
                            a = {'pos':(x,y), 'direc':1}
                        elif row == x and col == y - 2:
                            a = {'pos':(x,y), 'direc':2}
                        elif row == x and col == y + 2:
                            a = {'pos':(x,y), 'direc':3}
                        self.select['action'] = a 
                    
                    else:
                        if self.game.state['obs'][row, col] == 1:
                            self.select['pos'] = (row, col)
                            self.select['legal_pos'] = self.game.get_legal_pos((row, col))
                            print('legal position: ', self.select['legal_pos'])
                    
    
    def update(self):
        if time.time() - self.time < self.INTERVAL:
            return
        
        if self.human_mode == True:
            if self.select['action'] is not None:
                a = self.game.raw_to_std(self.select['action'])
                state, action, next_state, reward, done = self.game.step(a)
                # print(next_state)
                self.select = {'pos':None, 'legal_pos':[], 'action':None}
            
        if self.AI_mode == True and self.human_mode == False:
            if self.game.episodes > 0:
                if self.game.is_end() is not True:
                    # state (dict) : {'obs':obs, 'legal_actions':legal_actions} 
                    self.game.state['legal_actions'] = self.game.get_legal_actions(self.game.state)
                    # action = self.agent.main(self.game.state)
                    action = self.agent.step(self.game.state)
                    print("update ----> agent.step")
                    if action is not None:
                        state, action, next_state, reward, done = self.game.step(action)
                        if done:
                            # self.agent.feed_episode_ts(self.game.memory)
                            self.record_score(reward)
                            self.game.episodes -= 1
                            self.game.reset()
                    self.time = time.time()
            else:
                self.running = False
                
            # if self.max_length_track != []:
            #     action = self.max_length_track.pop(0)
            #     self.game.step(action)
            #     self.time = time.time()
            

    def render(self): 
        # Render background color
        self.window.fill(self.BgColor)
        
        # Render board background
        pygame.draw.rect(self.window, self.BoardColor, (0,0,self.SIZE*4,self.SIZE*4),0)
        
        # Render lines
        for i in range(self.ROW + 1):
            LINECOLOR = self.LineColor
            pygame.draw.line(self.window, LINECOLOR, (0, i*self.SIZE), (self.COL*self.SIZE, i*self.SIZE), 4)
        for i in range(self.COL + 1):
            pygame.draw.line(self.window, LINECOLOR, (i*self.SIZE, 0), (i*self.SIZE, self.ROW*self.SIZE), 4)
        
        # Render chess
        for i in range(self.ROW):
            for j in range(self.COL):
                if self.game.state['obs'][i, j] == 1:
                    self.window.blit(self.img, (8 + j*self.SIZE, 8 + i*self.SIZE))
        
        # Render selected effect
        if self.select['pos'] is not None:
            (x, y) = self.select['pos']
            self.window.blit(self.select_text, ((y+0.05)*self.SIZE , (x+0.5)*self.SIZE))
            # Render legal position when select a chocolate
            for one in self.select['legal_pos']:
                x, y = one
                self.window.blit(self.img_legal, ((y+0.3)*self.SIZE, (x+0.3)*self.SIZE))
            
        # Render MC tree            
        def plot_one_node(color, center, radius):
            pygame.draw.circle(self.window, color, center, radius)

        # print(f'------------->agent nodes length:{len(self.agent.nodes)}')
        for i, layer_nodes in enumerate(self.agent.nodes):
            for j, node in enumerate(layer_nodes):
                if hasattr(node, 'get_completed_visit_times') and node.get_completed_visit_times() == 0:
                    continue
                color = node.get_node_color()
                # center = (350+j*6, 65+i*20)
                center = (700+j*6, 65+i*20)
                radius = 3
                plot_one_node(color, center, radius)
        
        # Render Rollout times
        text = 'Rollout times: ' + str(150 - self.agent.rest_rollout_times)
        render_text = self.font.render(text, True, (0,0,0))
        # self.window.blit(render_text, (395, 20))
        self.window.blit(render_text, (705, 20))

        # pygame render refreshment 
        pygame.display.update()
        
    # main loop
    def run(self):
        self.render()
        while self.running:
            self.processInput()
            self.update()
            self.render()

            if self.AI_mode == False and self.human_mode == True:
                self.clock.tick(self.FPS)

        self.save_scores_as_csv()
        

###########################################################
if __name__ == '__main__':
    UI = UserInterface()
    UI.run()
    print('Game over!')
    pygame.quit()
    sys.exit()

