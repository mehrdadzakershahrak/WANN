import numpy as np
import os
import pickle
import config as run_config

# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD

# prettyNeat
from extern.wann.neat_src import * # NEAT and WANNs
from extern.wann.domain import *   # Task environments

import cloudpickle
from collections import defaultdict

log = run_config.log()
games = None

_ALG_CHECKPOINT_PATH=f'_checkpoint{os.sep}'
_ALG_CHECKPOINT_FN = f'alg-checkpoint.pkl'

def init_games_config(g):
  global games

  games = g

# -- Run NEAT ------------------------------------------------------------ -- #
def master(): 
  """Main NEAT optimization script
  """
  global fileName, hyp
  data = WannDataGatherer(fileName, hyp)

  if not os.path.exists(fileName+_ALG_CHECKPOINT_PATH):
    os.makedirs(fileName+_ALG_CHECKPOINT_PATH)

  if hyp['use_checkpoint']:
    with open(fileName+_ALG_CHECKPOINT_PATH+_ALG_CHECKPOINT_FN, 'rb') as f:
      alg = pickle.load(f)
  else:
    alg = Wann(hyp)

  for gen in range(hyp['maxGen']):
    if gen > 0:
      alg_critic = None
    else:
      alg_critic = hyp['alg_critic']

    pop = alg.ask()            # Get newly evolved individuals from NEAT
    reward = batchMpiEval(pop, alg_critic=alg_critic)  # Send pop to be evaluated by workers
    alg.tell(reward)           # Send fitness to NEAT    

    data = gatherData(data,alg,gen,hyp)
    print(gen, '\t', data.display())

    # checkpoint generation
    with open(fileName + _ALG_CHECKPOINT_PATH + _ALG_CHECKPOINT_FN, 'wb') as f:
      pickle.dump(alg, f, protocol=pickle.HIGHEST_PROTOCOL)

  # Clean up and data gathering at run end
  data = gatherData(data,alg,gen,hyp,savePop=True)
  data.save()
  data.savePop(alg.pop,fileName) # Save population as 2D numpy arrays


def gatherData(data,alg,gen,hyp,savePop=False):
  """Collects run data, saves it to disk, and exports pickled population

  Args:
    data       - (DataGatherer)  - collected run data
    alg        - (Wann)          - neat algorithm container
      .pop     - [Ind]           - list of individuals in population    
      .species - (Species)       - current species
    gen        - (ind)           - current generation
    hyp        - (dict)          - algorithm hyperparameters
    savePop    - (bool)          - save current population to disk?

  Return:
    data - (DataGatherer) - updated run data
  """
  data.gatherData(alg.pop, alg.species)
  if (gen%hyp['save_mod']) is 0:
    data = checkBest(data)
    data.save(gen)

  if savePop is True: # Get a sample pop to play with in notebooks    
    global fileName

    import pickle
    with open(fileName+'_pop.obj', 'wb') as fp:
      pickle.dump(alg.pop,fp)

  return data

def checkBest(data):
  """Checks better performing individual if it performs over many trials.
  Test a new 'best' individual with many different seeds to see if it really
  outperforms the current best.

  Args:
    data - (DataGatherer) - collected run data

  Return:
    data - (DataGatherer) - collected run data with best individual updated


  * This is a bit hacky, but is only for data gathering, and not optimization
  """
  global filename, hyp
  if data.newBest is True:
    bestReps = max(hyp['bestReps'], (nWorker-1))
    rep = np.tile(data.best[-1], bestReps)
    fitVector = batchMpiEval(rep, sameSeedForEachIndividual=False)
    trueFit = np.mean(fitVector)
    if trueFit > data.best[-2].fitness:  # Actually better!      
      data.best[-1].fitness = trueFit
      data.fit_top[-1]      = trueFit
      data.bestFitVec = fitVector
    else:                                # Just lucky!
      prev = hyp['save_mod']
      data.best[-prev:]    = data.best[-prev]
      data.fit_top[-prev:] = data.fit_top[-prev]
      data.newBest = False
  return data


# -- Parallelization ----------------------------------------------------- -- #
def batchMpiEval(pop, alg_critic=None, sameSeedForEachIndividual=True):
  """Sends population to workers for evaluation one batch at a time.

  Args:
    pop - [Ind] - list of individuals
      .wMat - (np_array) - weight matrix of network
              [N X N] 
      .aVec - (np_array) - activation function of each node
              [N X 1]

  Return:
    reward  - (np_array) - fitness value of each individual
              [N X 1]

  Todo:
    * Asynchronous evaluation instead of batches
  """
  global nWorker, hyp
  nSlave = nWorker-1
  nJobs = len(pop)
  nBatch= math.ceil(nJobs/nSlave) # First worker is master

  # Set same seed for each individual
  if sameSeedForEachIndividual is False:
    seed = np.random.randint(1000, size=nJobs)
  else:
    seed = np.random.randint(1000)

  if alg_critic is not None:
    msg = cloudpickle.dumps(alg_critic)

  reward = np.empty( (nJobs,hyp['alg_nVals']), dtype=np.float64)
  i = 0 # Index of fitness we are filling

  update_critic = defaultdict(lambda: True)
  for iBatch in range(nBatch): # Send one batch of individuals
    for iWork in range(nSlave): # (one to each worker if there)
      if i < nJobs:
        wVec   = pop[i].wMat.flatten()
        n_wVec = np.shape(wVec)[0]
        aVec   = pop[i].aVec.flatten()
        n_aVec = np.shape(aVec)[0]

        if alg_critic is not None and update_critic[iWork]:
          comm.send(1, dest=(iWork)+1, tag=7)
          comm.send(len(msg), dest=(iWork)+1, tag=6)
          comm.Send(msg, dest=(iWork) + 1, tag=6)
          update_critic[iWork] = False
        else:
          comm.send(0, dest=(iWork) + 1, tag=7)

        comm.send(n_wVec, dest=(iWork)+1, tag=1)
        comm.Send(  wVec, dest=(iWork)+1, tag=2)
        comm.send(n_aVec, dest=(iWork)+1, tag=3)
        comm.Send(  aVec, dest=(iWork)+1, tag=4)
        if sameSeedForEachIndividual is False:
          comm.send(seed.item(i), dest=(iWork)+1, tag=5)
        else:
          comm.send(  seed, dest=(iWork)+1, tag=5)        

      else: # message size of 0 is signal to shutdown workers
        n_wVec = 0
        comm.send(n_wVec,  dest=(iWork)+1)
      i = i+1 
  
    # Get fitness values back for that batch
    i -= nSlave
    for iWork in range(1,nSlave+1):
      if i < nJobs:
        workResult = np.empty(hyp['alg_nVals'], dtype='d')
        comm.Recv(workResult, source=iWork)
        reward[i,:] = workResult
      i+=1
  return reward

def slave():
  """Evaluation process: evaluates networks sent from master process. 

  PseudoArgs (recieved from master):
    wVec   - (np_array) - weight matrix as a flattened vector
             [1 X N**2]
    n_wVec - (int)      - length of weight vector (N**2)
    aVec   - (np_array) - activation function of each node 
             [1 X N]    - stored as ints, see applyAct in ann.py
    n_aVec - (int)      - length of activation vector (N)
    seed   - (int)      - random seed (for consistency across workers)

  PseudoReturn (sent to master):
    result - (float)    - fitness value of network
  """
  global hyp

  task = WannGymTask(games[hyp['task']], nReps=hyp['alg_nReps'])

  while True:
    # Evaluate any weight vectors sent this way
    # while True:
    update_critic = True if comm.recv(source=0, tag=7) == 1 else False
    if update_critic:
      n_alg_critic = comm.recv(source=0, tag=6)
      if n_alg_critic > 0:
        alg_critic = np.empty(n_alg_critic, dtype='d')
        comm.Recv(alg_critic, source=0, tag=6)
        alg_critic = cloudpickle.loads(alg_critic)
        hyp['alg_critic'] = alg_critic

    n_wVec = comm.recv(source=0,  tag=1)# how long is the array that's coming?
    if n_wVec > 0:
      wVec = np.empty(n_wVec, dtype='d')# allocate space to receive weights
      comm.Recv(wVec, source=0,  tag=2) # recieve weights

      n_aVec = comm.recv(source=0,tag=3)# how long is the array that's coming?
      aVec = np.empty(n_aVec, dtype='d')# allocate space to receive activation
      comm.Recv(aVec, source=0,  tag=4) # recieve it
      seed = comm.recv(source=0, tag=5) # random seed as int

      result = task.getFitness(wVec,aVec,hyp,seed=seed) # process it
      comm.Send(result, dest=0)            # send it back

      if n_wVec < 0: # End signal recieved
        break

def stopAllWorkers():
  """Sends signal to all workers to shutdown.
  """
  global nWorker
  nSlave = nWorker-1
  print('stopping workers')
  for iWork in range(nSlave):
    comm.send(-1, dest=(iWork)+1, tag=1)

# -- Input Parsing ------------------------------------------------------- -- #


def run(args, alg_critic, kill_slaves=False, use_checkpoint=False):
  if kill_slaves:
    stopAllWorkers()
    return

  global fileName, hyp, rank, nWorker # Used by both master and slave processes
  fileName    = args['outPrefix']
  hyp  = args['hyperparam']

  # TODO: clean this up HACK
  hyp['use_checkpoint'] = use_checkpoint
  hyp['alg_critic'] = alg_critic

  rank = args['rank']
  nWorker = args['num_workers']

  updateHyp(hyp, games)

  if (rank == 0):
    log.info('PERFORMING WANN TRAINING STEP...')
    master()
    log.info('PERFORMING WANN TRAINING STEP COMPLETE')
  else:
    slave()
    exit(0)


if __name__ == "__main__":
  run()
