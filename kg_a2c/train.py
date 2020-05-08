from tqdm import tqdm, trange
import argparse
from time import time

from custom_agent import CustomAgent
from environment import Environment

def get_points(score, wt):
    possible_points = get_points_from_wt(wt)

    percentage = (float(score) / possible_points) * 100
    return score, possible_points, percentage

def get_points_from_wt(wt):
    pts = []
    dropped_things = []
    for cmd in wt:
        cmd = cmd.replace('pork chop', 'pork c_op')
        if 'drop' in cmd:
            dropped_things.append(cmd.replace('drop ', ''))
            pts.append(0.0)
            continue
        if 'take' in cmd and not 'knife' in cmd and not any([thing in cmd for thing in dropped_things]):
            pts.append(1.0)
            continue
        if 'cook' in cmd:
            pts.append(1.0)
            continue
        if 'slice' in cmd or 'dice' in cmd or 'chop' in cmd:
            pts.append(1.0)
            continue
        if cmd in ['prepare meal', 'eat meal']:
            pts.append(1.0)
            continue
        pts.append(0.0)
    return int(sum(pts))

class Trainer:
    def __init__(self, game_dir):
        self.agent = CustomAgent()
        self.env = Environment(game_dir)

    def train(self):
        self.start_time = time()

        for epoch_no in tqdm(range(1, self.agent.nb_epochs + 1)):
            print('Epoch {}'.format(epoch_no))
            accuracy = 0.0
            for game_no in range(len(self.env.games)):
                obs, infos = self.env.reset()
                self.agent.train()

                scores = [0] * len(obs)
                dones = [False] * len(obs)
                steps = [0] * len(obs)
                while not all(dones):
                    # Increase step counts.
                    steps = [step + int(not done) for step, done in zip(steps, dones)]
                    commands = self.agent.act(obs, scores, dones, infos)
                    obs, scores, dones, infos = self.env.step(commands)

                # Let the agent know the game is done.
                self.agent.act(obs, scores, dones, infos)
                score = sum(scores) / self.agent.batch_size

                score, possible_points, percentage = get_points(score, infos['extra.walkthrough'][0])
                print('Score for game {}: {}/{}'.format(game_no+1, score, possible_points))
                accuracy += percentage

            print('Accuracy {}'.format(accuracy/len(self.env.games)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument('--games', action='store', type=str, help='Directory of the games used for training.')
    args = parser.parse_args()

    Trainer(args.games).train()