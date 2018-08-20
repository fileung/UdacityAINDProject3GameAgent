
from sample_players import DataPlayer
from isolation import Isolation, DebugState
import random
#from algorithms import alpha_beta_search, minimax, iterative_deepening
#from algorithms import MCTS

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def __init__(self, player_id):
        super().__init__(player_id)
        
        # notification - for dev only
        # log every time an instance of this class is created
        feedback ('-------------')        
        feedback ('---New Game---')        
        feedback ('--------------')        
 
        self.opening_book = self.load_opening_book()        

    def load_opening_book(self):
        # currently self.data is the opening book, but it could store more in the future
        # so keep self.data and self.opening_book seperated into two variables
        return self.data
    
    def get_opening_book_action(self, board):
        if board in self.opening_book:
            # notification - for dev only
            # print ('Opening Book found state = ', board)
            
            # each state only have only one action, the best action with the max reward
            return self.opening_book[board]
        else:
            return None
        
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
    
        action = None    
        on_depth = 0

        try:
#        if True:
            USE_OPENING_BOOK = True # switch this flag to get results for project report requirement - using opening book, and not using it
            VISUAL_OPENING_BOOK = False
            VISUAL_REST_OF_GAME = False

            # for the first 4 moves, use opening book if possible
            if USE_OPENING_BOOK and state.ply_count < 4:
                if VISUAL_OPENING_BOOK:
                    feedback ('You are player {0}, Move number {1}, Your turn to move! ({2})'.format(state.player()+1, state.ply_count+1, 'Opening Book'))
                    self.show_board(state)
                
                if self.opening_book is None:
                    feedback ('Opening Book does not exist!!')
                else:
                    action = self.get_opening_book_action(state.board)
            
            # if there is no move found in opening book, or number of moves played is greater than 4
            # use the default algorithm - minimax with iterative deepening, alpha beta pruning, or even mcts
            if action == None:                    
                if state.ply_count < 2:
                    if VISUAL_REST_OF_GAME:
                        feedback ('You are player {0}, Move number {1}, Your turn to move! ({2})'.format(state.player()+1, state.ply_count+1, 'Opening Move (Random)'))
                        self.show_board(state)

                    action = random.choice(state.actions())
                    
                else:
                    if VISUAL_REST_OF_GAME:
                        feedback ('You are player {0}, Move number {1}, Your turn to move! ({2})'.format(state.player()+1, state.ply_count+1, 'Iter Deep Alpha Beta Minimax'))
                        self.show_board(state)

                    # depth set to same as AI (depth=3) 
                    # to make sure its 'fair', as setting a depth more than AI will automatically be better, without any extra coding
                    depth = 3
                    
                    # iterative deepening, a move is guaranteed there will be an action in the queue, if depth=1 search is completed
                    # when time runs out, an exception will thrown, class 'isolation.StopSearch'     
                    # exception handling will put the best move to queue
                    # action = iterative_deepening(state, depth) # code changed not use from algorithms.py
                    
                    best_move = None
                    on_depth = 0
                    for d in range(1, depth+1):
    
                        # track dpeth
                        on_depth = d

                        # minimax - lower winning rate compare to alpha beta search    
                        # best_move = minimax(state, d)

                        # benchmark
#                        best_move = alpha_beta_search(state, d)
                        
                        mcts = MCTS(state)
                        best_move = mcts.Execute()
#                        print ('MCTS number of nodes created is {0}'.format(mcts.TreeNodeCOUNT))

                        action = best_move
    
            feedback('Action selected: {0}'.format(action))
            feedback()
            self.queue.put(action)
                                    
        except Exception as ex:
            # use best move when time runs out
            if str(type(ex)) == "<class 'isolation.StopSearch'>":
                feedback('Time runs out at depth={0}, iterative deepening best action is: {1}'.format(on_depth, action))
                feedback()
                if action is not None:
                    self.queue.put(action)
            
            feedback('Exception in get_action:')
            feedback('Type: '+str(type(ex)))
            feedback('Args: '+str(ex.args))
            feedback('Exception: '+str(ex))
            
            # pass the exception back to level above, the calling code
            raise ex 
            
    def show_board(self, state):
        dbstate = DebugState.from_state(state)
        feedback(dbstate) 
        
# a simple function to control print() statements by one flag
def feedback(print_text=''):
    _FEEDBACK = False
    if _FEEDBACK:
        print (print_text)
        

      
########################################################
### algorithms.py code have to move inside this file ### 
### otherwise udacity submit will not work           ### 
########################################################

####################################
###             MCTS             ###
####################################
import math

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
        epoch = 40 # for default time 150ms

        # project requirements - add more time
        # so increase epoch here to search more nodes, which use more time
        # epoch = 80 # for extra time 1000ms
        
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


"""
Algorithms - minimax, alpha beta pruning
All these code are all from lesson solutions, some properties are renamed

"""
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
    
    def min_value(state, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if state.terminal_test(): return state.utility(player_id)
        if depth <= 0: return score(state, player_id)
        
        v = float("inf")
        for a in state.actions():
            v = min(v, max_value(state.result(a), alpha, beta, depth-1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    def max_value(state, alpha, beta, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if state.terminal_test(): return state.utility(player_id)
        if depth <= 0: return score(state, player_id)
        
        v = float("-inf")
        for a in state.actions():
            v = max(v, min_value(state.result(a), alpha, beta, depth-1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    for a in state.actions():
        v = min_value(state.result(a), alpha, beta, depth-1)
        alpha = max(alpha, v)
        if v >= best_score:
            best_score = v
            best_move = a
    return best_move

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
    
