import random, util
from game import Agent
import numpy as np
from pacman import SCARED_TIME

#     ********* Reflex agent- sections a and b *********
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):

        """
           getAction chooses among the best options according to the evaluation function.

           getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
           ------------------------------------------------------------------------------
           """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
    The evaluation function takes in the current GameState (pacman.py) and the proposed action
    and returns a number, where higher numbers are better.
    """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return betterEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
  """
    return gameState.getScore()


######################################################################################
# b: implementing a better heuristic function
def GetFoodByDirection(game_state, pac_pos):

    food_grid = game_state.getFood()
    direction_food_dict = {a: 0 for a in ['East', 'West', 'North', 'South']}
    for x in range(food_grid.width):
        for y in range(food_grid.height):
            if food_grid.data[x][y]:
                if x > pac_pos[0]:
                    direction_food_dict['East'] += 1
                if x < pac_pos[0]:
                    direction_food_dict['West'] += 1
                if y > pac_pos[1]:
                    direction_food_dict['North'] += 1
                if y < pac_pos[1]:
                    direction_food_dict['South'] += 1

    return direction_food_dict

def betterEvaluationFunction(gameState):
    """

  The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

  A GameState specifies the full game state, including the food, capsules, agent configurations and more.
  Following are a few of the helper methods that you can use to query a GameState object to gather information about
  the present state of Pac-Man, the ghosts and the maze:

  gameState.getLegalActions():
  gameState.getPacmanState():
  gameState.getGhostStates():
  gameState.getNumAgents():
  gameState.getScore():
  The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
  """
    pac_pos = gameState.getPacmanPosition()
    curr_direction = gameState.getPacmanState().configuration.direction
    ghost_manhattan_dist = [util.manhattanDistance(pac_pos, ghost_pos) for ghost_pos in gameState.getGhostPositions()]
    capsules_manhattan_dist = [util.manhattanDistance(pac_pos, cap_pos) for cap_pos in gameState.getCapsules()]
    game_score = gameState.getScore()

    # food calculation
    if gameState.getNumFood() < 20:
        food_num_direction = GetFoodByDirection(gameState, pac_pos)
        food_num_curr_direction = food_num_direction[curr_direction]
    else:
        food_num_curr_direction = 0

    # ghost calculation
    ghost_manhattan_dist = [util.manhattanDistance(pac_pos, ghost_pos) for ghost_pos in gameState.getGhostPositions()]
    ghost_scared_time = [ghost.scaredTimer for ghost in gameState.getGhostStates()]
    scared_ghost_manhattan_dist = [ghost_dist for ghost_dist, time in zip(ghost_manhattan_dist, ghost_scared_time) if time > 0]
    min_scared_ghost_dist = min(scared_ghost_manhattan_dist) if len(scared_ghost_manhattan_dist) > 0 else 0
    min_ghost_dist = min(ghost_manhattan_dist)

    # capsule calculation
    capsule_eaten_now = all([time >= SCARED_TIME - 4 for time in ghost_scared_time])
    capsule_eaten = any(ghost_scared_time) and not capsule_eaten_now
    min_capsule_dist = min(capsules_manhattan_dist) if (len(capsules_manhattan_dist) > 0 and not capsule_eaten_now) else 0
    ghost_dist = min_scared_ghost_dist if capsule_eaten else min_ghost_dist

    # weight calculation
    ghost_weight = -5 if capsule_eaten else 3
    heuristic_weights = np.array([ghost_weight, -2, 0.1, 1])

    heuristic_params = np.array([ghost_dist, min_capsule_dist, food_num_curr_direction, game_score])
    score = heuristic_params @ heuristic_weights
    return score


#     ********* MultiAgent Search Agents- sections c,d,e,f*********

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def directionChangeNeed(self, current_state, next_state):
        pacman_pos = current_state.getPacmanPosition()
        ghost_dist = [util.manhattanDistance(pacman_pos, ghost_pos) for ghost_pos in current_state.getGhostPositions()]
        capsule_eaten_now = any([ghost.scaredTimer == SCARED_TIME for ghost in current_state.getGhostStates()])

        # if capsule_eaten and min(ghost_dist) > 2:
        #     return True
        if min(ghost_dist) < 2:
            return True

        food_dist = [util.manhattanDistance(pacman_pos, food_pos) for food_pos in GetFoodPosition(current_state)]
        capsules_dist = [util.manhattanDistance(pacman_pos, capsule_pos)
                         for capsule_pos in current_state.getCapsules()]
        if len(capsules_dist) == 0:
            capsules_dist = [np.inf]
        if (min(food_dist) > 1) and (min(capsules_dist) > 1):
            return True

        return False

def GetFoodPosition(gameState):
    FoodGrid = gameState.getFood()

    return [(x, y) for x in range(FoodGrid.width) for y in range(FoodGrid.height) if FoodGrid.data[x][y]]

######################################################################################
# c: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent
  """

    def getAction(self, gameState):

        """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
        legal_actions = gameState.getLegalActions(self.index)
        next_state_list = [gameState.generateSuccessor(self.index, a) for a in gameState.getLegalActions(self.index)]
        next_action_score = {a: self.RBminiMax(next_state, self.index, self.depth)
                             for a, next_state in zip(legal_actions, next_state_list)}

        best_score = max(next_action_score.values())
        best_action = [a for i, a in enumerate(legal_actions) if next_action_score[a] == best_score]

        action = np.random.choice(best_action)
        return action

    def RBminiMax(self, state, agent_idx, d):
        agent_idx = switch(state, agent_idx)
        legalMoves = state.getLegalActions(agent_idx)
        if state.isLose() or state.isWin() or len(legalMoves) == 0:
            return state.getScore()

        if d == 0:
            return self.evaluationFunction(state)

        if agent_idx == self.index: # agent pacman
            scores = (self.RBminiMax(state.generateSuccessor(agent_idx, a), agent_idx, d - 1) for a in legalMoves)
            return max(scores)

        else:
            scores = (self.RBminiMax(state.generateSuccessor(agent_idx, a), agent_idx, d) for a in legalMoves)
            return min(scores)


######################################################################################
# d: implementing alpha-beta
def switch(gameState, agent_idx):
    return (agent_idx + 1) % gameState.getNumAgents()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning
  """

    def getAction(self, gameState):
        """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

        # BEGIN_YOUR_CODE
        legal_actions = gameState.getLegalActions(self.index)
        alpha = -np.inf
        curr_max = -np.inf
        beta = np.inf
        best_action = 0

        for a in legal_actions:
            next_state = gameState.generateSuccessor(self.index, a)
            score = self.RBalphaBeta(next_state, self.index, self.depth, alpha, beta)
            if score > curr_max:
                curr_max = score
                alpha = score
                best_action = a

        return best_action

    def RBalphaBeta(self, state, agent_idx, d, alpha, beta):
        agent_idx = switch(state, agent_idx)
        legal_actions = state.getLegalActions(agent_idx)
        if state.isLose() or state.isWin() or len(legal_actions) == 0:
            return state.getScore()

        if d == 0:
            return self.evaluationFunction(state)

        if agent_idx == self.index:  # agent pacman
            curr_max = -np.inf
            for a in legal_actions:
                next_state = state.generateSuccessor(agent_idx, a)
                score = self.RBalphaBeta(next_state, agent_idx, d - 1, alpha, beta)
                curr_max = max(curr_max, score)
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return curr_max
            return curr_max

        else:
            curr_min = np.inf
            for a in legal_actions:
                next_state = state.generateSuccessor(agent_idx, a)
                score = self.RBalphaBeta(next_state, agent_idx, d, alpha, beta)
                curr_min = min(curr_min, score)
                beta = min(curr_min, beta)
                if curr_min <= alpha and agent_idx == (state.getNumAgents() - 1):
                    return curr_min
            return curr_min


######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent
  """

    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their legal moves.
    """

        legal_actions = gameState.getLegalActions(self.index)
        next_state_list = [gameState.generateSuccessor(self.index, a) for a in gameState.getLegalActions(self.index)]
        next_action_score = [self.RBrandomExpectimax(next_state, self.index, self.depth) for next_state in next_state_list]

        best_score = max(next_action_score)
        best_action = [a for i, a in enumerate(legal_actions) if next_action_score[i] == best_score]
        return np.random.choice(best_action)

    def RBrandomExpectimax(self, state, agent_idx, d):
        agent_idx = switch(state, agent_idx)
        legal_actions = state.getLegalActions(agent_idx)
        if state.isLose() or state.isWin() or len(legal_actions) == 0:
            return state.getScore()

        if d == 0:
            return self.evaluationFunction(state)

        if agent_idx == self.index:  # agent pacman
            scores = (self.RBrandomExpectimax(state.generateSuccessor(agent_idx, a), agent_idx, d - 1) for a in legal_actions)
            return max(scores)


        else:
            scores = np.array([self.RBrandomExpectimax(state.generateSuccessor(agent_idx, a), agent_idx, d)
                               for a in legal_actions])
            action_prob = np.array([1 / len(legal_actions)] * len(legal_actions))
            return scores @ action_prob


######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent
  """

    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
    """

        legal_actions = gameState.getLegalActions(self.index)
        next_state_list = [gameState.generateSuccessor(self.index, a) for a in gameState.getLegalActions(self.index)]
        next_action_score = [self.RBDirectionExpectimx(next_state, self.index, self.depth) for next_state in next_state_list]

        best_score = max(next_action_score)
        best_action = [a for i, a in enumerate(legal_actions) if next_action_score[i] == best_score]
        return np.random.choice(best_action)

    def RBDirectionExpectimx(self, state, agent_idx, d):
        agent_idx = switch(state, agent_idx)
        legal_actions = state.getLegalActions(agent_idx)
        if state.isLose() or state.isWin() or len(legal_actions) == 0:
            return state.getScore()

        if d == 0:
            return self.evaluationFunction(state)

        if agent_idx == self.index:  # agent pacman
            scores = (self.RBDirectionExpectimx(state.generateSuccessor(agent_idx, a), agent_idx, d - 1) for a in legal_actions)
            return max(scores)


        else:
            scores = np.array([self.RBDirectionExpectimx(state.generateSuccessor(agent_idx, a), agent_idx, d)
                               for a in legal_actions])
            action_prob = self.calcGhostDistribution(state, agent_idx)
            return scores @ np.array(list(action_prob.values()))

    def calcGhostDistribution(self, state, ghost_idx, prob_attack=0.8, prob_scared_flee=0.8):

        # Read variables from state
        ghost_state = state.getGhostState(ghost_idx)
        ghost_legal_actions = state.getLegalActions(ghost_idx)
        is_scared = ghost_state.scaredTimer > 0
        ghost_new_pos = [state.generateSuccessor(ghost_idx, a).getGhostPosition(ghost_idx) for a in ghost_legal_actions]

        pac_pos = state.getPacmanPosition()

        # Select best actions given the state
        distances_to_pacman = [util.manhattanDistance(pos, pac_pos) for pos in ghost_new_pos]

        if is_scared:
            best_score = max(distances_to_pacman)
            best_prob = prob_scared_flee
        else:
            best_score = min(distances_to_pacman)
            best_prob = prob_attack
        best_actions = [action for action, distance in zip(ghost_legal_actions, distances_to_pacman) if distance == best_score]

        # Construct distribution
        dist = util.Counter()
        for a in best_actions: dist[a] = best_prob / len(best_actions)
        for a in ghost_legal_actions: dist[a] += (1 - best_prob) / len(ghost_legal_actions)
        dist.normalize()
        return dist


######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
    """
    Your competition agent
  """
    def __init__(self):
        super().__init__()
        self.depth = 2

    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their legal moves.
    """

        legal_actions = gameState.getLegalActions(self.index)
        next_state_list = [gameState.generateSuccessor(self.index, a) for a in gameState.getLegalActions(self.index)]
        next_action_score = [self.RBrandomExpectimax(next_state, self.index, self.depth) for next_state in
                             next_state_list]

        best_score = max(next_action_score)
        best_action = [a for i, a in enumerate(legal_actions) if next_action_score[i] == best_score]
        return np.random.choice(best_action)

    def RBrandomExpectimax(self, state, agent_idx, d):
        agent_idx = switch(state, agent_idx)
        legal_actions = state.getLegalActions(agent_idx)
        if state.isLose() or state.isWin() or len(legal_actions) == 0:
            return state.getScore()

        if d == 0:
            return self.evaluationFunction(state)

        if agent_idx == self.index:  # agent pacman
            scores = (self.RBrandomExpectimax(state.generateSuccessor(agent_idx, a), agent_idx, d - 1) for a in
                      legal_actions)
            return max(scores)


        else:
            scores = np.array([self.RBrandomExpectimax(state.generateSuccessor(agent_idx, a), agent_idx, d)
                               for a in legal_actions])
            action_prob = np.array([1 / len(legal_actions)] * len(legal_actions))
            return scores @ action_prob