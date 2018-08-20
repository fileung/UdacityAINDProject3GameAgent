#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Udacity AIND Project 3 Game Agent - Opening Book
Build a opening book, with no time limit, and overwrite data pickle file
"""
from isolation import Isolation #, DebugState
from collections import defaultdict, Counter
import random
import time
import pickle
    
def main():

    start_time = time.time()
    
    state = Isolation()
    book = OpeningBook(initial_state=state, num_rounds=99*98*8*8*(10), tree_depth=4).get_book()
    OpeningBook.save_opening_book(book)
    
    end_time = time.time()
    # time taken to build opening book =  332.32017993927
    
    print ('time taken to build opening book = ', end_time - start_time)

class OpeningBook():

    def __init__(self, initial_state, num_rounds, tree_depth):
        
        # properties
        self.state = initial_state
        self.NUM_ROUNDS = num_rounds
        self.tree_depth = tree_depth
        
        # build
        self.table = self.build_table(self.NUM_ROUNDS)
               
    def get_book(self):
        return self.table
    
    def save_opening_book(book):
        with open('data.pickle', 'wb') as f:
            pickle.dump(book, f)
            
    def build_table(self, num_rounds):
        # Builds a table that maps from game state -> action
        # by choosing the action that accumulates the most
        # wins for the active player. (Note that this uses
        # raw win counts, which are a poor statistic to
        # estimate the value of an action; better statistics
        # exist.)
        book = defaultdict(Counter)
    
        for _ in range(num_rounds):
            
            # progress checking, feedback every 1000 rounds
            if _ % 1000 == 0:
                print ('round=', _)
            
            # new blank state for each round
            state = Isolation()        

            self.build_tree(state, book, self.tree_depth)
            
        # this line chooses the max score action of each board state, 
        # effectively removing all other possible actions of the board state which scored less
        return {k: max(v, key=v.get) for k, v in book.items()}
    
    
    def build_tree(self, state, book, depth):
        
        # if depth reach sepcified limit, the simulate the rest of the game by random moves
        if depth <= 0 or state.terminal_test():
            # evaluate by simulate the rest of the game by random actions
            # return -self.simulate(state) 
            # evaulate by having more possible moves
            return -self.score(state) 
        
        # random move to explore
        action = random.choice(state.actions())
        
        # recursivly get reward from deeper tree state-action
        reward = self.build_tree(state.result(action), book, depth - 1)

        # accumulate reward from current state of the board, for this action
        book[state.board][action] += reward

        # negative as last state action (level above) is opponents move 
        return -reward 
    
    # evaluation function of opening book from lesson, play the game randomly to the end
    def simulate(self, state):
        player_id = state.player()
        while not state.terminal_test():
            state = state.result(random.choice(state.actions()))
        return -1 if state.utility(player_id) < 0 else 1
    
    # evaluation function used by AI minimax - score by diff in number of possible moves
    def score(self, state):
        player_id = state.player()
        own_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        diff = len(own_liberties) - len(opp_liberties)
#        return -1 if diff < 0 else 1
        return diff
    
if __name__ == "__main__":
    main()