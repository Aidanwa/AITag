from tag_env import TagEnv




def randomRun():
    env = TagEnv(grid_size=5)

    # Example game loop
    obs, _ = env.reset()
    steps = 0
    terminated, truncated = False, False
    while not terminated and not truncated:
        env.render()

        # Random actions for tagger and runner
        tagger_action = env.action_space[0].sample()
        runner_action = env.action_space[1].sample()
        print(f"Runner Action: {runner_action}\nTagger Action: {tagger_action}")

        obs, rewards, terminated, truncated, info = env.step((tagger_action, runner_action))
        steps += 1
        print(f"Rewards: {info}\n")
        print(f'Step: {steps}')

if __name__ == "__main__":
    randomRun()
