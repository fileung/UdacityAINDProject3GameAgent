#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Udacity AIND Project 3 Game Agent - Opening Book
Analysis matches.log
extraxt 1st and 2nd moves cell position
for reporting, the display output needs to able to easily copy + paste all records into multiple cells in spreadsheet, not line by line individually
format: gameindex tab move1 tab move2
"""

def read_logs():
    with open('matches.log', 'r') as f:
        txt = f.read()
        f.close()
    return txt

def write_stats(text):
    with open('stats_1_2_moves.txt', 'a') as f:
        f.write(text)
        f.write('\n')
        f.close()
        
def clear_stats():
    with open('stats_1_2_moves.txt', 'w') as f:
        f.write('')
        f.close()

logs = read_logs()

tag_new_game = 'INFO:isolation:Initial game state: Isolation'
games = logs.split(tag_new_game)

# check
# print (games[0])

# determine which player is SELF
#tag1 = 'First agent: Agent('
#tag2 = 'Second agent: Agent('
tag_self_agent_is_player1_A = "First agent: Agent(agent_class=<class 'my_custom_player.CustomPlayer'>, name='Custom Agent')"
tag_self_agent_is_player1_B = "First agent: Agent(agent_class=<class 'my_custom_player.CustomPlayer'>, name='Custom TestAgent')"
tag_self_agent_won = "Winner: Agent(agent_class=<class 'my_custom_player.CustomPlayer'>"
tag_self_agent_lost = "Loser: Agent(agent_class=<class 'my_custom_player.CustomPlayer'>"
tag_history_start = 'History: ['
tag_history_end = ']'

clear_stats()

#write_stats ('{0}\t{1}\t{2}\t{3}'.format('game', 'player1', 'move 1', 'move 2', 'won'))
write_stats ('{0}\t{1}\t{2}\t{3}'.format('player1', 'm1', 'm2', 'won'))
for i, game_text in enumerate(games):
            
    self_is_player_1 = False
    history_found = False

    # make sure i am on text where i can retrieve history
    try:
        game_text.index(tag_history_start)
        history_found = True
    except:
        history_found = False
        
    try:
        game_text.index(tag_self_agent_is_player1_A)
        self_is_player_1 = True
    except:        
        try:
            game_text.index(tag_self_agent_is_player1_B)
            self_is_player_1 = True
        except:
            self_is_player_1 = False
        
    
        
    player1 = None
    if self_is_player_1 == True:
#        print ('self Custom Agent = player 1')
        player1 = 'Self'
    else:
#        print ('self Custom Agent = player 2')
        player1 = 'AI'
    
    if history_found:
        start_crop = game_text.index(tag_history_start) + len(tag_history_start)
        end_crop = game_text.index(tag_history_end)
        history_inner_text = game_text[start_crop:end_crop]
        actions = history_inner_text.split(',')
#        print (history_text)

        self_agent_won = -1 # default value, -1 cannot happen in match results, but if self VS self then the result cannot be zero
        try:
            if game_text.index(tag_self_agent_won):
                self_agent_won = 1 # win value
        except: 
            do_nothing = ''
        
        try:
            if game_text.index(tag_self_agent_lost):
                self_agent_won = 0 # loss value
        except:
            do_nothing = ''
            
        if self_agent_won == -1:
            self_agent_won = 'N/A'

        # game number, [tab], move 1, [tab], move 2
        write_stats ('{0}\t{1}\t{2}\t{3}'.format(player1, actions[0], actions[1], self_agent_won))
#        write_stats ('{0}\t{1}\t{2}\t{3}\t{4}'.format(i, player1, actions[0], actions[1], self_agent_won))
    




###############
# testing
#exit

#aaa = 'abcefg'
#print (aaa.index('b'))
#print (aaa[2:3])
