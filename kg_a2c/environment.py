import os

import gym
import textworld.gym
from textworld import EnvInfos

class Environment:
    def __init__(self, games_dir, max_nb_steps=100, batch_size=1):
        self.games = self.fetch_games(games_dir)
        self.max_nb_steps = max_nb_steps
        self.batch_size = batch_size
        self.env = self.setup()

    def fetch_games(self, games_dir):
        games = []
        for game in os.listdir(games_dir):
            if game.endswith(".ulx"):
                games.append(os.path.join(games_dir, game))

        print("Found total {} games".format(len(games)))
        return games

    def setup(self) -> EnvInfos:
        requested_infos = EnvInfos()
        requested_infos.description = True
        requested_infos.inventory = True
        requested_infos.entities = True
        requested_infos.verbs = True
        requested_infos.extras = ["recipe", "walkthrough"]
        requested_infos.admissible_commands = True

        env_id = textworld.gym.register_games(self.games,requested_infos, max_episode_steps=self.max_nb_steps, name="training")
        env_id = textworld.gym.make_batch(env_id, batch_size=self.batch_size, parallel=True)

        return gym.make(env_id)

    def step(self, commands):
        return self.env.step(commands)

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def play_manually(self):
        try:
            done = False
            self.reset()
            nb_moves = 0
            while not done:
                self.render()
                command = input("Enter command:")
                nb_moves += 1
                obs, scores, dones, infos = self.step([command])

        except KeyboardInterrupt:
            pass

        print("Total {}  steps taken, scored {} points".format(nb_moves,scores[0]))


if __name__ == "__main__":
    env = Environment(games_dir="../sample_games")
    env.play_manually()