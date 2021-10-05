import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd

#### generate a set of pokers

class Poker(object):
    '''
    Poker class contianing a set of pokers
    '''
    type_mapping = {1:'Heart',2:'Tile',3:'Clover',4:'Pike'}
    values = {1:'Ace', 11:'J', 12:'Q',13:'K'}
    def __init__(self, n=1):

        '''
        Given number of poker sets, initialize all poker cards
        '''
        self.num_sets = n
        self.cards_available = np.array(range(n*52))
        self.cards_used = np.ones(n*52,dtype=bool)

    def draw_card(self,num_cards=1, translate = False):
        '''
        Draw Nnum_cards from the current deck
        :param num_cards:
        :param translate:
        :return:
        '''
        if np.all(self.cards_used==False):
            ### all cards are draw in this deck
            raise ValueError('Cards all distributed. Inititate a new class')
        ids = np.random.choice(self.cards_available[self.cards_used], size = num_cards, replace=False)
        ### rebalance the internal records

        self.cards_used[ids] = False ### mask these data points
        if translate:
            return np.array([self.translate_card(x) for x in ids])
        else:
            return ids


    def translate_card(self, id):

        loc1 = id%52
        type_id = loc1//13 + 1
        loc2 = loc1%13 + 1
        return Poker.type_mapping[type_id]+'_'+Poker.values.get(loc2, str(loc2))

class BlackJack_Dealer(Poker):
    '''
    Class of a BalckJack_Dealer
    '''
    def __init__(self, n_player = 1, number_of_decks = 2):
        '''
        Initiate the Game
        :param n_player: Number of players in the game. The minimum is 1. We will always take idx=0
        :param number_of_decks: Number of poker decks used in each game.
        '''
        ### We will be the 0th player
        self.n_player = n_player
        self.players = {x:-1*np.ones(2) for x in range(n_player)} ###initial all cards to -1
        self.dealer = -1*np.ones(2) ### set all cards to -1
        self.points = {x:0 for x in range(n_player)}
        self.points[-1] = 0 ### idx -1 represents dealer
        self.player_bets = np.zeros(n_player)
        self.player_bust = np.zeros(n_player) ### 0, go-on, 1: win, exit, -1: lost
        self.insurance_option = False
        self.player_insurance = {x: False for x in range(n_player)}
        super().__init__(number_of_decks)

    def start_game(self, initial_bet = 10, enable_strategy = False, print_msg = True):
        '''
        Draw Cards for dealer and each players,
        :param initial_bet: our initial bets in the game
        :param enable_strategy: indicator to activate our simulation strategy
        :param print_msg: Boolean to turn-off print function for key messages to speed up simulation if necessary
        :return: our profit from the game as player0
        '''

        for i in range(self.player_bets.shape[0]):
            if i==0:
                self.player_bets[i] = -1*initial_bet ### we mark it negative here to indicate players are giving away money
            else:
                ### if there are other players, randomly assign initial bets between 1 to 100
                self.player_bets[i] = -1*np.random.randint(1, 100)

        '''
        Time to draw cards for each player and the dealer
        '''
        ### two cards to Dealer
        self.dealer = self.draw_card(2) ### draw two cards for dealer

        ### for each player, draw 2 cards,
        ### cards are distributed in sequency
        for i in self.players:
            self.players[i] = self.draw_card(2)


        ### calculate the score of each player
        self.cal_score()
        ### show the current holdings of dealer(masked) and players
        if print_msg:
            self.print_holding()

        ### check if dealer has aces or 10, if so, insurance option will be enabled
        ### since the player can only see the second card, we will only check the second card of the dealer
        if self.dealer[1]%13+1==1 or self.dealer[1]%13+1>=10:
            ### dealer has ace or card value at 10
            print(f'Dealer First Card is {self.translate_card(self.dealer[1])}')

            ### default insurance to True
            self.insurance_option = True
            for idx in self.player_insurance:
                ### TODO: Add logic to consider insurance
                ### By default, all other players are
                self.player_insurance[idx] = True
                self.player_bets[idx] = self.player_bets[idx]*1.5


        if 21 in self.points[-1]:
            if print_msg:
                print('Dealer Check Second Card. BlackJack! Anyone without BlackJack or Insurance lose their bets.')
            self.print_holding(review_dealer = True)
            for idx in self.points:
                if idx==-1:
                    continue
                else:
                    ### check if a player has 21, if not lost
                    if self.player_insurance[idx]:
                        if print_msg:
                            print(f"Player{idx} has insurance. Get 2X bets")
                        self.player_bust[idx] = 1 ### win (insurance)
                        self.player_bets[idx] = -1.5*self.player_bets[idx] ### insruance portion payout 2:1, original bet lost
                    elif 21 in self.points[idx]:
                        if print_msg:
                            print(f"Player{idx} hits BlackJack. It's a pull. Return Original Bets")
                        self.player_bust[idx] = 1 ### win (pull)
                        self.player_bets[idx] = -1*self.player_bets[idx]
                    else:
                        if print_msg:
                            print(f"Player{idx} lost.")
                        self.player_bust[idx] = -1 ### lost
                        self.player_bets[idx] = 0 ### no payout at all, 0 return
            return self.player_bets[0], None ### return my bets, terminate the game
        else:
            if print_msg:
                print("Dealer Check Second Card. Not BlackJack. Players' turn.")
            ### check if player has blackjack
            for idx in self.points:
                if 21 in self.points[idx]:
                    ### this player hits BlackJack
                    if print_msg:
                        print(f"Player{idx} hits blackJack. Win 1.5X the bets")
                    self.player_bets[idx] = -2.5*self.player_bets[idx] #### payout rate 1.5:1, so total payout 2.5
                    self.player_bust[idx] = 1
            if self.player_bust[0]==1:
                ### We win
                return self.player_bets[0], None
            ### dealer has no BlackJack neither do we, game go-on
            ### our position is 0, we will be the first to execute
            if enable_strategy:
                if print_msg:
                    print('Strategy Enabled')
                ### We know all the cards by each player and one card from dealer.
                ### we calcualte the probability of getting a number larger than dealer
                ### based on current score, check the probabaly of drawing one card and our score is higher than dealer

                ### gather information on cards draw (2 for each player and 1 from dealer)
                used_card = np.array([self.players[x] for x in range(self.n_player)]).ravel()
                used_card = np.append(used_card,self.dealer[1]) ### we don;t know index=0 delaer card
                total_available_cards = set(range(self.num_sets*52)) - set(used_card)
                prob = simulate_prob(my_cards = self.players[0], dealer_card=self.dealer[1], available_cards = np.array(list(total_available_cards)))
                action = np.argmax(prob)
                if action == 0:
                    if print_msg:
                        print('Simulation shows that it is best to take no action')
                    pass
                elif action == 1:
                    if print_msg:
                        ### draw 1 card
                        print('Simuation shows it is best to draw 1 card')
                    self.action_hit(0)
                else:
                    if print_msg:
                        ### draw 2 cards
                        print('Simuation shows it is best to draw 2 card')
                    self.action_hit(0)
                    self.action_hit(0)
            else:
                ### by default, action is random
                for i in range(self.n_player):
                    action = np.random.choice([0,1,2])
                    if print_msg:
                        print(f'Player{i} decide to draw {action} cards')
                    if action == 0:
                        pass
                    else:
                        for _ in range(action):
                            ### draw cards
                            self.action_hit(i)

        ### Dealer to review its cards, check if anyone lost
        if print_msg:
            print("Dealer Review its Cards")
            self.print_holding(review_dealer = True)

        ### dealer to hit till reach score above 21
        while np.all(self.points[-1]<17):
            ### if current dealer score is smaller than 17
            ### dealer continue to draw cards
            self.action_hit(-1)

        if np.any(self.points[-1]<=21):
            ### after dealer hits, its total score is below 21
            ### we need to check if
            dealer_score = np.max(self.points[-1][self.points[-1]<=21]) ###

            for idx in self.points:
                if idx==-1:
                    continue
                if abs(self.player_bust[idx])>0.5:
                    ### player idx busted before this round
                    continue
                elif np.min(self.points[idx])>21:
                    ### busted
                    self.player_bust[idx] = -1 ### lost
                    self.player_bets[idx] = 0 ### no payout at all, 0 return
                    if print_msg:
                        print(f"Player{idx} busted with score {np.min(self.points[idx])}")
                else:
                    p_score = np.max(self.points[idx][self.points[idx]<=21])

                    if dealer_score == p_score:
                        ### push
                        self.player_bust[idx] = 1 ### win (pull)
                        self.player_bets[idx] = -1*self.player_bets[idx]
                    elif dealer_score>p_score:
                        ### dealer win
                        self.player_bust[idx] = -1 ### lost
                        self.player_bets[idx] = 0 ### no payout at all, 0 return
                        if print_msg:
                            print(f"Player{idx} busted since dealer score {dealer_score} is higher than player score {p_score}")
                    else:
                        ### dealer<p_score
                        if p_score==21:
                            self.player_bust[idx] = 1 ### win (BackJack)
                            self.player_bets[idx] = -2.5*self.player_bets[idx] ### payout 1.5:1
                            if print_msg:
                                print(f"Player{idx} hits blackJack. Win 1.5X the bets")
                        else:
                            self.player_bust[idx] = 1 ### win (BackJack)
                            self.player_bets[idx] = -2*self.player_bets[idx] ### payout 1.5:1
                            if print_msg:
                                print(f"Player{idx} win (without BlackJack) with score {p_score} while dealer score is {dealer_score}")
            ### return our bets
            return self.player_bets[0], action
        else:
            ### dealer busted
            for idx in self.points:
                if idx==-1:
                    continue
                if np.min(self.points[idx])==21:
                    self.player_bust[idx] = 1 ### win (BackJack)
                    self.player_bets[idx] = -2.5*self.player_bets[idx] ### payout 1.5:1
                    if print_msg:
                        print(f"Player{idx} hits blackJack. Win 1.5X the bets")
                elif np.min(self.points[idx])<21:
                    self.player_bust[idx] = 1 ### win (BackJack)
                    self.player_bets[idx] = -2*self.player_bets[idx] ### payout 1.5:1
                    if print_msg:
                        print(f"Player{idx} win (without BlackJack) with score {np.min(self.points[idx])} since dealer busted.")
                else:
                    ### dealer win
                    self.player_bust[idx] = -1 ### lost
                    self.player_bets[idx] = 0 ### no payout at all, 0 return
                    if print_msg:
                        print(f"Player{idx} busted with socre {np.min(self.points[idx])}")

            ### return our bets
            return self.player_bets[0], action

    def action_hit(self, idx):
        '''
        Given the index of the player, draw one additional card
        :param idx: player id, -1 is dealer
        :return:
        '''

        new_card = self.draw_card(num_cards = 1)
        print(f"New Card is {self.translate_card(new_card[0])}")
        ### add to current player holding
        if idx == -1:
            self.dealer = np.append(self.dealer, new_card[0])
        else:
            self.players[idx] = np.append(self.players[idx], new_card[0])
        ### recalcualte the score for this player
        self.cal_score(idx)


    def cal_score(self, idx = None):
        '''
        Calculate the score
        :param idx: if None, calculate score for all, if number si given, calculate that player
        :return:
        '''
        if idx is None:
            for i in self.points:
                if i == -1:
                    cards = self.dealer
                else:
                    cards = self.players[i]
                self.points[i] = cal_score_np(cards)

        else:
            if idx == -1:
                cards = self.dealer
            else:
                cards = self.players[idx]

            self.points[idx] = cal_score_np(cards)


    def print_holding(self, review_dealer = False):
        '''
        Print the cards of each player
        :param review_dealer: if True, the second card of the player will be revealed
        :return:
        '''

        deal_s = 'Dealer   '
        for idx, card in enumerate(sorted(self.dealer)):
            if review_dealer == False and idx==0:
                deal_s += '****  '
            else:
                deal_s += self.translate_card(card)+'  '
        print(deal_s)

        for p_idx, cards in self.players.items():
            if p_idx == 0:
                player_s = 'You      '
            else:
                player_s = f'Player{p_idx}  '
            for card in cards:
                player_s += self.translate_card(card)+'  '
            print(player_s)

@njit
def cal_score_np(score_array):
    '''

    :param score_array: numpy array, holdings
    :return:
    '''

    loc1 = score_array%52
    loc2 = np.minimum(loc1%13 + 1,10)
    score = np.sum(loc2)
    ### count how many 1 in the score
    res = np.array([score])
    for num in loc2:
        if num == 1:
            score += 10
        res = np.append(res,score)

    return res

@njit
def dealer_win(d_score, my_score):

    '''
    Gicen dealer score and my_score, return True if dealer win
    :param d_score: np.array
    :param my_score: np.array
    :return: True if dealer win
    '''

    if np.all(d_score>21) and np.all(my_score>21):
        ### both busted
        return False
    elif np.all(d_score>21) and (not np.all(my_score>21)):
        ### delaer busted,I am okay
        return False
    elif (not np.all(d_score>21)) and np.all(my_score>21):
        return True
    else:
        d_score_max = np.max(d_score[d_score<=21])
        my_score_max = np.max(my_score[my_score<=21])
        if d_score_max>my_score_max:
            return True
        else:
            return False

@njit
def simulate_prob(my_cards, dealer_card, available_cards):
    '''
    calculate the prob of winning by simulation
    Three scenario are simulated, and the highest win prob action is return:
    1. I take no action
    2. I draw a card
    3. I draw 2 cards

    :param my_cards: 2-element array indicating my card
    :param dealer_card: int, indicating dealer's card
    :param available_cards: all cards not drawn (plus hidden card of dealer)
    :return: prob. of our score is higher than dealer and not busted
    '''

    prob = np.zeros(3, dtype=np.float64)
    my_init_score = cal_score_np(my_cards)
    n_sim = 10000


    ### simulate Case One:
    ### I stand, no action
    win = 0
    for _ in range(n_sim):
        used_cards = np.full(available_cards.shape[0], True)
        dealer_hidden_card = np.random.choice(available_cards)
        used_cards[np.where(available_cards==dealer_hidden_card)] = False

        ### I take no action, dealer keeps drawing till score reached >=17
        d_holding = np.array([dealer_card,dealer_hidden_card])
        d_score = cal_score_np(d_holding)
        while (np.max(d_score)<17) and (np.max(d_score[d_score<=21])<np.max(my_init_score[my_init_score<=21])):
            ### assume dealer will draws card if below 17 or score lower than mine
            ### draw another card
            another_card = np.random.choice(available_cards[used_cards])
            used_cards[np.where(available_cards==another_card)] = False
            d_holding = np.append(d_holding, another_card)
            d_score = cal_score_np(d_holding)
        ### judge who win
        if not dealer_win(d_score, my_init_score):
            win +=1

    prob[0] = win/n_sim

    ### simulate case two:
    ### I draw one card
    win = 0
    for _ in range(n_sim):
        my_holding = np.copy(my_cards)
        used_cards = np.full(available_cards.shape[0], True)
        dealer_hidden_card = np.random.choice(available_cards)
        used_cards[np.where(available_cards==dealer_hidden_card)] = False
        d_holding = np.array([dealer_card,dealer_hidden_card])
        d_score = cal_score_np(d_holding)

        ### I draw one card
        my_third_card = np.random.choice(available_cards[used_cards])
        used_cards[np.where(available_cards==my_third_card)] = False
        my_holding = np.append(my_holding, my_third_card)
        my_score = cal_score_np(my_holding)
        while (np.max(d_score)<17) and (np.min(my_score)>21 or np.max(d_score[d_score<=21])<np.max(my_score[my_score<=21])):
            ### assume dealer will draws card if below 17 or score lower than mine
            ### draw another card
            another_card = np.random.choice(available_cards[used_cards])
            used_cards[np.where(available_cards==another_card)] = False
            d_holding = np.append(d_holding, another_card)
            d_score = cal_score_np(d_holding)

        ### judge who win
        if not dealer_win(d_score, my_score):
            win +=1

    prob[1] = win/n_sim

    ### simulation case three
    ### I draw 2 cards
    win = 0
    for _ in range(n_sim):
        my_holding = np.copy(my_cards)
        used_cards = np.full(available_cards.shape[0], True)
        dealer_hidden_card = np.random.choice(available_cards)
        used_cards[np.where(available_cards==dealer_hidden_card)] = False
        d_holding = np.array([dealer_card,dealer_hidden_card])
        d_score = cal_score_np(d_holding)

        ### I draw two card
        my_third_card = np.random.choice(available_cards[used_cards])
        used_cards[np.where(available_cards==my_third_card)] = False
        my_fourth_card = np.random.choice(available_cards[used_cards])
        used_cards[np.where(available_cards==my_fourth_card)] = False

        my_holding = np.append(my_holding, np.array([my_third_card, my_fourth_card]))
        my_score = cal_score_np(my_holding)

        while (np.max(d_score)<17) and (np.min(my_score)>21 or np.max(d_score[d_score<=21])<np.max(my_score[my_score<=21])):
            ### assume dealer will draws card if below 17 or score lower than mine
            ### draw another card
            another_card = np.random.choice(available_cards[used_cards])
            used_cards[np.where(available_cards==another_card)] = False
            d_holding = np.append(d_holding, another_card)
            d_score = cal_score_np(d_holding)

        ### judge who win
        if not dealer_win(d_score, my_score):
            win +=1
    prob[2] = win/n_sim

    return prob


if __name__ =='__main__':

    ### test Poker Class
    # poker = Poker(n=1)
    # print(poker.draw_card(2))
    # print(poker.draw_card(2))
    # print(poker.draw_card(2))
    print('Test the simulator')
    game = BlackJack_Dealer(n_player = 3, number_of_decks = 2)
    initial_bet = 10
    profit, action = game.start_game(initial_bet, enable_strategy = False, print_msg = True)
    print(f'Test game got return of {profit} with action {action}.')

    num_games=1000
    random_returns = -2*np.ones(num_games)
    random_action = -1*np.ones(num_games)
    strategy_return = -2*np.ones(num_games)
    strategy_action = -1*np.ones(num_games)
    np.random.seed(1)
    ### test simulation
    ### run 1000 game with 10 dollar bet,record the return
    for idx in range(num_games):
        game = BlackJack_Dealer(n_player = 3, number_of_decks = 2)
        initial_bet = 10
        profit, action = game.start_game(initial_bet, enable_strategy = False, print_msg = False)
        if game.player_insurance[0]:
            initial_bet = initial_bet*1.5
        return_rate = profit/initial_bet - 1
        random_returns[idx] = return_rate
        random_action[idx] = action

    np.random.seed(1)
    for idx in range(num_games):
        game = BlackJack_Dealer(n_player = 3, number_of_decks = 2)
        initial_bet = 10
        profit, action = game.start_game(initial_bet, enable_strategy = True, print_msg = False)
        if game.player_insurance[0]:
            initial_bet = initial_bet*1.5
        return_rate = profit/initial_bet - 1
        strategy_return[idx] = return_rate
        strategy_action[idx] = action

    # fig, axes = plt.subplots()
    # axes.plot(range(1, num_games+1), random_returns, label = 'Random_Return', alpha = 0.5, color= 'lightcoral')
    # axes.plot(range(1, num_games+1), random_returns, label = 'Strategy_Return', alpha = 0.5, color = 'lightsteelblue')
    # axes.set_title('Black Jack Return')
    # axes.set_xlabel('Num of Simulated Games')
    # axes.set_ylabel('Return Rate')
    # plt.show()



    analysis = pd.DataFrame(index = range(num_games), columns = ['random_return', 'strategy_return', 'random_action', 'strategy_action'])

    analysis['random_return'] = random_returns
    analysis['strategy_return'] = strategy_return
    analysis['random_action'] = random_action
    analysis['strategy_action'] = strategy_action

    analysis.to_excel('Black_Jack_Simulation.xlsx', engine = 'xlsxwriter')