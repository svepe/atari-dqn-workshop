import cv2
import numpy as np
import cupy as cp
import pandas as pd

from chainer import Chain, cuda
from chainer import links as L
from chainer import functions as F
from chainer.optimizers import Adam
from chainer.serializers import save_npz, load_npz


from gym import make


class ExperienceBuffer:
    def __init__(self, length, state_shape):
        self.length = length
        self.index = 0

        self.states = np.zeros((length,) + state_shape, np.float32)
        self.actions = np.zeros((length,), np.int)
        self.rewards = np.zeros((length,), np.float32)
        self.next_states = np.zeros((length,) + state_shape, np.float32)
        self.terminals = np.zeros((length,), np.float32)

    def append(self, state, action, reward, next_state, terminal):
        self.states[self.index] = state
        self.actions[self.index] = int(action)
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.terminals[self.index] = float(terminal)

        self.index += 1
        if self.index == self.length:
            self.index = 0

    def sample(self, n):
        idxs = np.random.choice(self.length, n, False).tolist()
        return (
            cp.asarray(self.states[idxs]),
            cp.asarray(self.actions[idxs]),
            cp.asarray(self.rewards[idxs]),
            cp.asarray(self.next_states[idxs]),
            cp.asarray(self.terminals[idxs])
        )


class DQN(Chain):
    def __init__(self, num_actions):
        super().__init__(
            l1=L.Convolution2D(4, 32, 8, 4),
            l2=L.Convolution2D(32, 64, 4, 2),
            l3=L.Convolution2D(64, 64, 3, 1),
            l4=L.Linear(512),
            lQ=L.Linear(num_actions))

    def __call__(self, state):
        h1 = F.relu(self.l1(state))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        return self.lQ(h4)


class Agent:
    def __init__(self, game, train=True):
        self.env = make(game)

        # num_frames x height x width
        self.state_shape = (4, 84, 84)

    def test(self, episodes, filename="pong_dqn.npz"):
        dqn = DQN(self.env.action_space.n).to_gpu()
        load_npz(filename, dqn)

        epsilon = 0.01

        for e in range(episodes):
            state = np.zeros(self.state_shape, np.float32)

            frame = self.env.reset()
            state = self.update_state(frame, state)

            total_reward = 0.

            done = False
            while not done:
                if np.random.rand() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    Qs = dqn(cp.asarray(state[np.newaxis]))
                    action = cp.argmax(Qs.array).get()

                frame, reward, done, info = self.env.step(action)
                next_state = self.update_state(frame, state)

                state = next_state

                total_reward += reward
                self.env.render()

            print("Episode {}: {}".format(e, total_reward))

    def train(self, episodes):
        initial_exploration_period = 10**5
        experience_buffer = ExperienceBuffer(10**5, self.state_shape)
        iteration = 0

        dqn = DQN(self.env.action_space.n).to_gpu()
        target_dqn = DQN(self.env.action_space.n).to_gpu()
        min_epsilon = 0.1
        epsilon_step = 1e-6
        epsilon = 1.
        save_period = 2 * 10**4

        log = pd.DataFrame(
            columns=["iteration", "total_reward", "average_loss", "epsilon"]
        )

        optimizer = Adam(alpha=1e-4)
        optimizer.setup(dqn)

        for e in range(episodes):
            state = np.zeros(self.state_shape, np.float32)

            frame = self.env.reset()
            state = self.update_state(frame, state)

            total_reward = 0.
            episode_duration = 0
            total_loss = 0.

            done = False
            while not done:
                if np.random.rand() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    Qs = dqn(cp.asarray(state[np.newaxis]))
                    action = cp.argmax(Qs.array).get()

                next_frame, reward, done, info = self.env.step(action)
                next_state = self.update_state(next_frame, state)

                experience_buffer.append(
                    state, action, reward, next_state, done
                )

                state = next_state

                if iteration > initial_exploration_period:
                    total_loss += self.replay(
                        experience_buffer, dqn, target_dqn, iteration, optimizer
                    )
                    epsilon = max(min_epsilon, epsilon - epsilon_step)

                    if iteration % save_period == 0:
                        print("Saved.")
                        save_npz("pong_dqn.npz", dqn)
                        log.to_pickle("training_log.p")

                total_reward += reward
                iteration += 1
                episode_duration += 1

            average_loss = total_loss / episode_duration
            log.loc[e] = [iteration, total_reward, average_loss, epsilon]
            print(
                "Episode {0:} / {1:}: {2:0.8f}, {3:}".format(
                    e, iteration, average_loss, total_reward
                )
            )

    def replay(self, experience_buffer, dqn, target_dqn, iteration, optimizer):
        batch_size = 32
        target_dqn_update_period = 10**4
        discount_factor = 0.99

        batch = experience_buffer.sample(batch_size)
        states, actions, rewards, next_states, terminals = batch

        target_Qsa = cp.max(target_dqn(next_states).array, 1)
        y = rewards + discount_factor * target_Qsa * (1 - terminals)
        Qsa = dqn(states)[cp.arange(batch_size), actions]

        loss = F.sum(F.squared_error(y, Qsa)) / batch_size

        dqn.cleargrads()
        loss.backward()
        optimizer.update()

        if iteration % target_dqn_update_period == 0:
            target_dqn.copyparams(dqn)

        return loss.array.get()

    def update_state(self, frame, state):
        return np.vstack((state[1:], self.preprocess(frame)[np.newaxis]))

    def preprocess(self, frame):
        # Convert to grayscale
        res = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Scale frame by x0.525 from 210x160 to 110x84
        res = cv2.resize(res, None, fx=0.525, fy=0.525)
        # Crop 84x84 square image and normalise
        res = res[18:-8, :] / 255.0
        return res.astype(np.float32)


with cuda.get_device(0):
    agent = Agent("PongNoFrameskip-v4")
    agent.train(1000)
