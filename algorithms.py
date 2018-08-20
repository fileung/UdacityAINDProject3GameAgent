#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithms - minimax, alpha beta pruning
All these code are all from lesson solutions, some properties are renamed

"""
####################################
###             MCTS             ###
####################################
import random
import math

# debug imports
from isolation import Isolation, DebugState
state = Isolation()

class MCTS():

    def __init__(self, state):
        self.root_node = self.TreeNode(0, state)
        self.TreeNodeCOUNT = 1
         
    #####################################################
    # MCTS 4 steps
    # 1.Select - select a unexplored child, or best child
    # 2.Expand - pick an action, execute and get next child
    # 3.Simulate - simulate to the end of the game, get reward win=1, loss=-1
    # 4.Backpropagation - update all nodes with reward, from leaf node all the way back to the root
    #####################################################
    
    # MCTS step 1 of 4
    def select(self, node):    
        # loop from root to leaf node, keep selecting best node on each level
        # 1. at start, there is only one node, the root, to select
        #    once the root level is passed, select the best children 
        # 2. expand a node, by using an available action
        # 3. once all actions is executed, then select the best child by score

        while not node.state.terminal_test():
            self.feedback('select.while.nextnode.nodeid = {0}'.format(node.id))
            if not node.explored():
                self.feedback('select.expand()')   
                expand_node = self.expand(node)
                
                self.feedback('expand_node selected, the board state:')
                self.show_board(expand_node.state)
            
                return expand_node
            else:
                node = self.best_child(node)
            
            if node is None:
                self.feedback('select.best_child=NONE')
            else:
                self.feedback('select.best_child.nodeid={0}, q-value={1}'.format(node.id, node.q_value))
                
        return node
    
    # MCTS part of step 1 of 4
    def best_child(self, node):

        best_child_nodes = []
        best_score = float('-inf')

        # explore constant C
        C = 0.5 #math.sqrt(2)
        for child in node.childrens:
            # score math formula from wikipedia
            # children node score = exploit + explore
            #   exploit = wins / node visited count
            #   explore = explore factor * square_root( log(total number of simulation) / node visited count )
            exploit = child.q_value / child.visited
            explore = C * math.sqrt( math.log(node.visited) / child.visited )
            child_score = exploit + explore
            
            if child_score == best_score:
                best_child_nodes.append(child)
                
            elif child_score > best_score:
                best_child_nodes = []
                best_child_nodes.append(child)
                best_score = child_score
            
        # print ('best_child() best_child_nodes=', best_child_nodes[0])

        if len(best_child_nodes) == 0:
            self.feedback('best_child() found 0 best child!')
            self.show_board(node.state)
            self.feedback('node.childrens = {0}'.format(len(node.childrens)))
            return None

        # must select randomly from list of equally best childrens
        # otherwise the first children always get selected, and the tree node reward get skewed / not balanced
        return random.choice(best_child_nodes) 

    # MCTS step 2 of 4        
    def expand(self, node):
        # run next action, add next child to node children list 
        
        possible_actions = node.actions_available()
        
        if len(possible_actions) > 0:
            action = possible_actions[0]
            
            # action result state
            child_state = node.state.result(action)
            
            # node add a new child node
            child_node = MCTS.TreeNode(self.TreeNodeCOUNT, child_state, node, action)
            self.TreeNodeCOUNT += 1
            node.childrens.append(child_node)
            node.actioned.append(action)
            
            self.feedback('expand() new node created, nodeid = {0}'.format(child_node.id))

            # return child just added to the end of the list
            return node.childrens[-1]
        else:
            return None

    # MCTS step 3 of 4
    def simulate(self, state):
        player_id = state.player()
        while not state.terminal_test():
            state = state.result(random.choice(state.actions()))
        return -1 if state._has_liberties(player_id) else 1

    # MCTS step 4 of 4
    def backpropagation(self, node, reward):
        while node is not None:
            node.update_qvalue(reward)
            node = node.parent
            reward = -reward
    
    # best action is the best child parent action
    def best_action(self, node):
        return self.best_child(node).parent_action
        
    # execute MCTS and find best game state action
    def Execute(self):
                
        # final tuned epoch 
        # any higher could cause timeout, which loses the game
        epoch = 40 
        
        try:
            self.feedback('**********')
            self.feedback('***MCTS***')
            
            # code copy to try make thing work
            if self.root_node.state.terminal_test():
                return random.choice(self.root_node.state.actions())
        
            for i in range(epoch):
                
                if i % 10 == 0:
                    self.feedback('executing {0}'.format(i))
                        
                node = self.select(self.root_node)
                
                if node is None:
                    continue
                
                reward = self.simulate(node.state)
                self.backpropagation(node, reward)
                            
        except Exception as ex:            
            # must print any exception, even feedback is turned off
            print ('Exception: {0}'.format(str(ex)))
                        
        action = self.best_action(self.root_node)
        
        self.feedback ('---root to all childrens---')
        self.display_node_childs(self.root_node, 0)
        self.feedback ('-----')
        
        return action
        
    # show board visually
    def show_board(self, state):
        dbstate = DebugState.from_state(state)
        self.feedback(dbstate) 

    # list node children's childrens to the end leaf node
    def display_node_childs(self, node, level):
        self.feedback('level={0}, id={1}, q-value={2}'.format(level, node.id, node.q_value))
        for c in node.childrens:
            self.display_node_childs(c, level+1)
            
    # a simple function to control print() statements by one flag
    def feedback(self, print_text=''):
        _FEEDBACK = False
        if _FEEDBACK:
            print (print_text)
    
    # Node class
    # represents a game state, with available actions to explore future game states further down the tree
    class TreeNode():
        def __init__(self, nodeid, state, parent=None, parent_action=None):
            
            # game state
            self.state = state 
            
            # parent node, root have none
            self.parent = parent 

            # the parent action resulted this state, if this node is selected as best child with best score
            self.parent_action = parent_action

            # all actions of current node state
            self.actions = state.actions() 
            
            # applied actions
            self.actioned = [] 
            
            # store children nodes, contains result game states by actions
            self.childrens = [] 
            
            # accumulative reward, the name borrowed from reinforcement learning
            self.q_value = 0 
            
            # number of times this node been simulated 
            self.visited = 1 
            
            # id, useful to keep track of a tree node during dev / debug
            self.id = nodeid
 
        # keep track of current node childrens are 100% explored
        # return true when all state actions are explored
        def explored(self):
            self.feedback('explored actions={0} actioned={1}'.format(len(self.actions), len(self.actioned)))
            return len(self.actions) == len(self.actioned)
        
        # list the remaining action to be use to the rest of the node childrens
        def actions_available(self):
            actions_left = list(set(self.actions) - set(self.actioned))
            self.feedback('actions diff available = {0}'.format(actions_left))
            return actions_left
        
        def update_qvalue(self, reward):
            self.q_value += reward
            self.visited += 1
            
        # control print statements using a centralised flag, for debug only
        def feedback(self, print_text=''):
            _FEEDBACK = False
            if _FEEDBACK:
                print (print_text)


####################################
# 6.2.17 Coding: Iterative_Deepening
# using iterative_deepening
# search depth=1 first, when depth=1 finishes, then search on depth=2, then depth=3 etc
# this guarantee a move is available before time runs out
def iterative_deepening(state, depth):
    best_move = None
    for d in range(1, depth+1):
        # best_move = minimax(state, depth)
        best_move = alpha_beta_search(state, d)
        
        # trace the depth been executed
        # print ('iterative deepening next depth =', d)
    return best_move

# 6.2.25 Coding: Alpha Beta Pruning
# with code merging with minimax
def alpha_beta_search(state, depth):
    """ Return the move along a branch of the game tree that
    has the best possible value.  A move is a pair of coordinates
    in (column, row) order corresponding to a legal move for
    the searching player.
    
    You can ignore the special case of calling this function
    from a terminal state.
    """
    
    player_id = state.player()
    
    alpha = float("-inf")
    beta = float("inf")
    best_score = float("-inf")
    best_move = None
    
    total_nodes_count = 0

    # function signature to accept an alpha and beta parameter
    def min_value(state, alpha, beta, depth, nodes_count):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if state.terminal_test(): return state.utility(player_id)
        if depth <= 0: return score(state, player_id)
        
        v = float("inf")
        for a in state.actions():
            v = min(v, max_value(state.result(a), alpha, beta, depth-1, nodes_count))
            nodes_count += 1
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    # function signature to accept an alpha and beta parameter
    def max_value(state, alpha, beta, depth, nodes_count):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if state.terminal_test(): return state.utility(player_id)
        if depth <= 0: return score(state, player_id)
        
        v = float("-inf")
        for a in state.actions():
            v = max(v, min_value(state.result(a), alpha, beta, depth-1, nodes_count))
            nodes_count += 1
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
    
    for a in state.actions():
        v = min_value(state.result(a), alpha, beta, depth-1, total_nodes_count)
        alpha = max(alpha, v)
        if v > best_score:
            best_score = v
            best_move = a
    return best_move, total_nodes_count

# AI minimax, from this project itself, in sample_players.py
def minimax(state, depth):

    player_id = state.player()
    
    def min_value(state, depth):
        if state.terminal_test(): return state.utility(player_id)
        if depth <= 0: return score(state, player_id)
        value = float("inf")
        for action in state.actions():
            value = min(value, max_value(state.result(action), depth - 1))
        return value

    def max_value(state, depth):
        if state.terminal_test(): return state.utility(player_id)
        if depth <= 0: return score(state, player_id)
        value = float("-inf")
        for action in state.actions():
            value = max(value, min_value(state.result(action), depth - 1))
        return value

    return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))

def score(state, player_id):
    own_loc = state.locs[player_id]
    opp_loc = state.locs[1 - player_id]
    own_liberties = state.liberties(own_loc)
    opp_liberties = state.liberties(opp_loc)
    return len(own_liberties) - len(opp_liberties)
    
        
        
