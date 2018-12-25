import random, util
from game import Agent
import numpy as np


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
    ghost_mahattan_dist = [util.manhattanDistance(pac_pos, ghost_pos) for ghost_pos in gameState.getGhostPositions()]
    capsules_manhattan_dist = [util.manhattanDistance(pac_pos, cap_pos) for cap_pos in gameState.getCapsules()]

    min_ghost_dist = min(ghost_mahattan_dist) if len(ghost_mahattan_dist) > 0 else 0
    min_capsule_dist = min(capsules_manhattan_dist) if len(capsules_manhattan_dist) > 0 else 0
    food_num = gameState.getNumFood()
    min_food_dist = min([util.manhattanDistance(pac_pos, food_pos) for food_pos in gameState.getFood()])
    game_score = gameState.getScore()
    ghost_scared_time = [ghost.scaredTimer for ghost in gameState.getGhostStates()]

    capsule_eaten = any(ghost_scared_time)
    ghost_weight = -2 if capsule_eaten else 2
    capsule_weight = -1 if capsule_eaten else 2

    heuristic_weights = np.array([ghost_weight, capsule_weight, -2 * (food_num < 10), 1])
    heuristic_params = np.array([min_ghost_dist, min_capsule_dist, min_food_dist, game_score])
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


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
        next_action_score = [self.RBminiMax(next_state, self.index, self.depth) for next_state in next_state_list]

        best_score = max(next_action_score)
        best_action = [a for i, a in enumerate(legal_actions) if next_action_score[i] == best_score]
        return np.random.choice(best_action)

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
            return scores @ action_prob

    def calcGhostDistribution(self, state, ghost_idx):
        legal_actions = state.getLegalActions(ghost_idx)
        return np.array([1 / len(legal_actions)] * len(legal_actions))
    '''
        # Read variables from state
        ghostState = state.getGhostState(ghost_idx)
        legalActions = state.getLegalActions(ghost_idx)
        pos = state.getGhostPosition(ghost_idx)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5


        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += (1 - bestProb) / len(legalActions)
        dist.normalize()
        return dist
    '''

######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
    """
    Your competition agent
  """

    def getAction(self, gameState):
        """
      Returns the action using self.depth and self.evaluationFunction

    """

        # BEGIN_YOUR_CODE
        raise Exception("Not implemented yet")
        # END_YOUR_CODE
