# -*- coding: utf-8 -*-
import numpy as np
import random


def is_array_in_list(np_array, array_list):
    return list(np_array.flatten()) in [list(array.flatten()) for array in array_list]


def get_child_nodes_color(action_num) -> list:
    Red = (255, 0, 24)
    Green = (0, 128, 24)
    Blue = (92, 125, 237)
    Orange = (255, 165, 44)
    Purple = (134, 0, 125)
    Yellow = (255, 255, 65)
    Basic_Color = (Green, Red, Blue, Orange, Purple, Yellow)
    
    if action_num <= 6:
        color_list = [Basic_Color[i] for i in range(action_num)]
        return color_list
    else:
        color_list = list(Basic_Color)
        for i in range(action_num - 6):
            random_color = (random.choice(range(1,6))*50, 
                            random.choice(range(1,6))*50, 
                            random.choice(range(1,6))*50)
            while random_color in color_list:
                random_color = (random.choice(range(1,6))*50, 
                                random.choice(range(1,6))*50, 
                                random.choice(range(1,6))*50)
            color_list.append(random_color)
        return color_list