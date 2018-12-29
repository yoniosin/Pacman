from layout import getLayout
from pacman import *
from ghostAgents import *
from submission import OriginalReflexAgent, ReflexAgent, MinimaxAgent, AlphaBetaAgent, RandomExpectimaxAgent, DirectionalExpectimaxAgent
from textDisplay import *
# Multiprocessing
from multiprocessing import Process, Lock
from itertools import product

withDepthplayers = [DirectionalExpectimaxAgent]
# withDepthplayers = [MinimaxAgent, AlphaBetaAgent, RandomExpectimaxAgent]
withoutDepthplayers = []
# withoutDepthplayers = [OriginalReflexAgent, ReflexAgent]
depths = [4]
# depths = [2, 3, 4]
layouts = ['trickyClassic']
# layouts = ['capsuleClassic', 'contestClassic', 'mediumClassic',
#            'minimaxClassic', 'openClassic', 'originalClassic',
#            'smallClassic', 'testClassic', 'trappedClassic', 'trickyClassic']
ghosts = [DirectionalGhost(1), DirectionalGhost(2)]


def processesFunction(lock, player, layout_name, filename, depth=1):
    print("!!!!!!!!!!!!!!!!!another process!!!!!!!!!!!!")

    layout = getLayout(layout_name)
    if depth > 1:
        player.depth = depth

    games = runGames(layout, player, ghosts, NullGraphics(), 5, False, 0, False, 30)
    scores = [game.state.getScore() for game in games]
    times = [game.my_avg_time for game in games]
    avg_score = sum(scores) / float(len(scores))
    avg_time = sum(times) / float(len(times))
    line = (player.__class__.__name__ + ',' +
            str(depth) + ',' +
            layout_name + ',' +
            '%.2f' % avg_score + ',' +
            '%.2f' % (avg_time * 1e6) + 'E-06\n')
    
    # Begin of critical code
    lock.acquire()
    try:
        with open(filename, 'a') as file_ptr:
            file_ptr.write(line)
        file_ptr.close()
    finally:
        lock.release()

    # End of critical code
    return

if __name__ == '__main__':
    base = time.time()
    runs = []
    lock = Lock()

    # An array for the processes.
    processing_jobs = []

    for layout in layouts:
        filename = 'results_' + layout + '.csv'
        if os.path.exists(filename):
            os.remove(filename)

    for layout in layouts:
        filename = 'results_' + layout + '.csv'
        file_ptr = open(filename, 'w+')
        file_ptr.close()

    for layout in layouts:
        filename = 'results_' + layout + '.csv'
        for player in withoutDepthplayers:
            runs.append((lock, player(), layout, filename))
    for d in depths:
        for player in withDepthplayers:
            for layout in layouts:
                runs.append((lock, player(), layout, filename, d))
    
    print("total number of runs: ", len(runs))

    numOfCPUs = os.cpu_count()
    print("numOfCPUs: ", numOfCPUs)
    
    totalCyclesNum = int(len(runs) / numOfCPUs)
    print("totalCyclesNum: ", totalCyclesNum)

    for cycleNum in range(totalCyclesNum):
        processing_jobs = []

        print("you have only ", len(runs) - cycleNum * numOfCPUs," runs to go (*7) (!!!DONT WORRY BE HAPPY!!!)")
        print("Starting new cycle of process, cycle num: ", totalCyclesNum)

        for numberOfProcessInCycle in range(numOfCPUs):
            print("running run number: ", numberOfProcessInCycle + cycleNum * numOfCPUs)
            p = Process(target=processesFunction, args=runs[numberOfProcessInCycle + cycleNum * numOfCPUs])
            processing_jobs.append(p)
            p.start()
        
        for proc in processing_jobs:
            proc.join()
            
        # Empty active job list.
        del processing_jobs[:] 

    processing_jobs = []

    lastCycleSize = len(runs) - totalCyclesNum * numOfCPUs
    
    for nummberOfProcessInCycle in range(lastCycleSize):
        p = Process(target=processesFunction, args=runs[len(runs) - nummberOfProcessInCycle - 1])
        processing_jobs.append(p)
        p.start()
    
    for proc in processing_jobs:
        proc.join()

    # Empty active job list.
    del processing_jobs[:] 

    print('experiments time: ', (time.time() - base)/60, 'min')
    
