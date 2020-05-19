import numpy
class WAN(object):  #Weight Agnostic Neural
    def __init__(self, init_shared_weight):
        self.num_hidden = 10
        self.input_size = 5
        self.output_size = 1
        self.shape_in = [self.input_size, self.num_hidden]
        self.shape_out = [self.num_hidden, self.output_size]
        self.aVec = [1,1,1,1,1,1,7,7,5,1,5,5,4,1,7,3,9,1,3,7,9,5,4,3,9,7,1,7,1]
        self.wKey = [10,35,36,41,64,69,95,97,108,125,128,142,157,202,231,257,289,302,331,361,362,363,364,367,368,373,374,376,394,395,398,401,403,425,461,484,517,543,574,576,602,603,604,606,633,662,692,722,723,753,782,811]
        self.weights = [-0.1783,-0.0303,1.5435,1.8088,-0.857,1.024,-0.3872,0.2639,-1.138,-0.2857,0.3797,-0.199,1.3008,-1.4126,-1.3841,7.1232,-1.5903,-0.6301,0.8013,-1.1348,-0.7306,0.006,1.4754,1.1144,-1.5251,-1.277,1.0933,0.1666,-0.5483,2.6779,-1.2728,0.4593,-0.2608,0.1183,-2.1036,-0.3119,-1.0469,0.2662,0.7156,0.0328,0.3441,-0.1147,-0.0553,-0.4123,-3.2276,2.5201,1.7362,-2.9654,0.9641,-1.7355,-0.1573,2.9135]
        self.weight_bias = -1.5
        nNodes = len(self.aVec)
        self.wVec = [0] * (nNodes*nNodes)
        for i in range(nNodes*nNodes):
            self.wVec[i] = 0
        self.set_weight(init_shared_weight, 0)

    def set_weight(self, weight, weight_bias):
        nValues = len(self.wKey)
        if type(weight_bias).__name__ not in ['int','long','float']:
            weight_bias = 0
        if type(weight).__name__ == 'list':
            weights = weight
        else:
            weights = [weight] * nValues
        for i in range(nValues):
            k = self.wKey[i]
            self.wVec[k] = weights[i] + weight_bias

    def tune_weights(self):
        self.set_weight(self.weights, self.weight_bias)

    def get_action(self, old_state):
        nNodes = len(self.aVec)
        wMat = numpy.array(self.wVec).reshape((nNodes, nNodes))
        nodeAct = [0] * nNodes
        nodeAct[0] = 1
        for i in range(len(old_state)):
            nodeAct[i+1] = old_state[i]
        for iNode in range(self.input_size+1, nNodes):
            rawAct = numpy.dot(nodeAct, wMat[:, iNode:iNode+1])  #TPJ
            rawAct = self.applyActSimple(self.aVec[iNode], rawAct.tolist()[0])
            nodeAct[iNode] = rawAct
        return nodeAct[-self.output_size:][0]

    def applyActSimple(self, actId, x):
        if actId == 1:
            return x
        elif actId == 2:
            return 0.0 if x<=0.0 else 1.0  #unsigned step
        elif actId == 3:
            return math.sin(math.pi*x)
        elif actId == 4:  
            return math.exp(-(x*x)/2.0)  #gaussian with mean zero and unit variance 1
        elif actId == 5:
            return math.tanh(x)
        elif actId == 6:
            return (math.tanh(x/2.0) + 1.0)/2.0  #sigmoid
        elif actId == 7:
            return -x
        elif actId == 8:
            return math.abs(x)
        elif actId == 9:
            return max(x, 0)  #relu
        elif actId == 10:
            return math.cos(math.pi*x)
        else:
            print('unsupported actionvation type: ',actId)
            return None

def drl(environment):
    environment = CartPoleEnv()  #import gym environment = gym.make('CartPole-v0')
    drl = DRL(environment.observation_space.shape[0], 256, environment.action_space.shape[0])  #.n
    for epoch in range(1000):
        state_old = environment.reset()
        rewards = 1
        while True:
            environment.render()
            action_now = drl.explore(state_old)            
            state_new, reward_now, done, _ = environment.step(action_now)
            if done:
                reward_now = -1
            drl.remember(state_old, action_now, reward_now, state_new)
            if done:
                break
            rewards += reward_now
            state_old = state_new
            drl.rethink()   #TPJ: the chance to rethink is very import: not-done
        print('epoch=%04d'%(epoch),'  ','rewards=%d'%(rewards))

def tpj():
    #1.) Initialize：Create population of minimal networks.
    #2.) Evaluate：Test with range of shared weight values.
    #3.) Rank：Rank by performance and complexity 
    #4.) Vary：Create new population by varying best networks.
    #TODO

def wan():
    environment = CartPoleSwingUpEnv()
    drl = WAN(-1.5)
    for epoch in range(20):
        if epoch == 0:
            print('init_weights:')
        elif epoch == 10:
            print()
            print('tune_weights:')
            drl.tune_weights()

        state_old = environment.reset()
        rewards = 1
        for step in range(100000000):
            environment.render()
            action_now = drl.get_action(state_old)
            state_new, reward_now, done = environment.step(action_now)
            if done:
                reward_now = -1
                break
            rewards += reward_now
            state_old = state_new
        print('epoch=%04d'%(epoch),'  ','rewards=%d'%(rewards),'  ','step=%d'%(step))

if __name__ == '__main__':    
    #drl()
    #wan()
    tpj()
