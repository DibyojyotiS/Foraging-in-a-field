import gym
import berry_field

class DQN():
    def init(self, env, gamma, epsilon, tau, bufferSize, updateFrequency, MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES, explorationStrategyTrainFn,
    explorationStrategyEvalFn, optimizerFn):

        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.bufferSize = bufferSize
        self.updateFrequency = updateFrequency
        self.MAX_TRAIN_EPISODES = MAX_TRAIN_EPISODES
        self.MAX_EVAL_EPISODES = MAX_EVAL_EPISODES
        self.explorationStrategyTrainFn = explorationStrategyTrainFn
        self.explorationStrategyEvalFn = explorationStrategyEvalFn
        self.optimizerFn = optimizerFn


        self.nnTarget = createQNetwork(S, A, hidDim, activationFunction)
        self.nnOnline = createQNetwork(S, A, hidDim, activationFunction)
        copyNetworks(self.nnOnline, self.nnTarget)
        self.initBookeeping()
        self.rBuffer = ReplayBuffer(bufferSize)

    def runDQN():
        resultTrain = self.trainAgent()
        resultsEval = self.evaluateAgent()
        plotResults()
        return result, final_eval_score, training_time, wallclock_time

    def trainAgent(self):
        copyNetworks(self.nnOnline, self.nnTarget)
        for e in range(self.MAX_TRAIN_EPISODES):
            s, done = self.env.reset()
            self.rBuffer.collectExperiences(self.env, s, self.ExplorationStrategyTrainFn)
            experiences = self.rBuffer.sample(batchSize)
            self.trainQN(experiences)
            self.performBookeeping(train = True)
            self.evaluateAgent(qNetwork, self.MAX_EVAL_EPISODES)
            self.performBookeeping(train=False)
            if e%self.updateFrequency == 0:
                copyNetwork(nnOnline, nnTarget)
    
    def trainQN(self, experiences):
        ss, a, rs, sNexts, dones = self.rBuffer.splitExperiences(experiences)
        max_a_qs = self.nnTarget(sNexts).detach().max()
        tdTargets = rs + self.gamma * max_a_qs * (1 - dones)
        qs = self.nnOnline(ss).gather(a)
        tdErrors = tdTargets - qs
        loss = mean(0.5*(tdErrors)**2)
        optimizerFn.init()
        loss.backward()
        optimizerFn.step()

    def EvaluateAgent(self, qNetwork, MAX_EVAL_EPISODES):
        rewards = []
        for e in range(MAX_EVAL_EPISODES):
            rs = 0
            s, done = self.env.reset()
            for c in count():
                a = self.explorationStrategyEvalFn(nnOnline, s)
                s, r, done = self.env.step(a)
                rs += r
                if done:
                    rewards.append(rs)
                    break
        self.performBookeeping(train=False)
        return mean(rewards), std(rewards)

    def initBookKeeping(self):
        pass
    
    def performBookeeping(train=True):
        pass

class ReplayBuffer():
    def init(self, bufferSize):
        pass
    def store(self, experience):
        pass
    def length(self):
        pass

    def collectExperiences(self, env, s, explorationStrategy):
        pass

    def sample(self, batchSize):
        pass

    def splitExperiences(self, experiences):
        pass
