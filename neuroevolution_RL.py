import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# create neural network agent


def create_agent():
    model = Sequential()
    model.add(Dense(128, input_shape=(4,), activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])
    return model

# create multiple random nets


def return_random_agents(number):
    agents = []
    for _ in range(number):

        agent = create_agent()
        agents.append(agent)

    return agents

# test agents


def run_agent(model, runs, env):
    scores = []
    sim_steps = 500
    SHOW_EVERY = 100
    for _ in range(runs):
        observation = env.reset()
        score = 0
        for step in range(sim_steps):
            action = np.argmax(model.predict(observation.reshape(1, 4)))
            observation, reward, done, _ = env.step(action)

            if step % SHOW_EVERY == 0:
                env.render()
            score += reward
            if done:
                break
        scores.append(score)
    print(np.mean(scores))
    return np.mean(scores)

# consolidate agent scores


def return_agent_stackscores(agents, runs, env):
    stackscore = []
    for agent in agents:
        stackscore.append(run_agent(agent, runs, env))
    return stackscore

# reproduction function


def reproduce(agentX, agentY):

    child_agent = create_agent()
    for i, layer in enumerate(child_agent.layers):
        x_weight = agentX.layers[i].get_weights()
        y_weight = agentY.layers[i].get_weights()
        new_weight = (np.array(x_weight) + np.array(y_weight))/2
        for i in range(len(new_weight)):
            new_weight[i] += np.random.randn() * 0.02
        layer.set_weights(new_weight)

    return child_agent

# reproduce top agents


def return_children(agents, sorted_parent_indexes):

    children_agents = []

    for i in range(len(agents)-1):
        selectedX_agent_index = sorted_parent_indexes[np.random.randint(
            len(sorted_parent_indexes))]
        selectedY_agent_index = sorted_parent_indexes[np.random.randint(
            len(sorted_parent_indexes))]
        children_agents.append(
            reproduce(agents[selectedX_agent_index], agents[selectedY_agent_index]))

    return children_agents

# Main run


num_agents = 50
agents = return_random_agents(num_agents)

# How many top agents to consider as parents
top_limit = 10

# run evolution until X generations
generations = 5

elite_index = None

env = gym.make("CartPole-v0")

for generation in range(generations):

    # return rewards of agents
    rewards = return_agent_stackscores(
        agents, 5, env)  # return average of 50 runs

    # sort by rewards
    # reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
    sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit]
    print("")
    print("")

    top_rewards = []
    for best_parent in sorted_parent_indexes:
        top_rewards.append(rewards[best_parent])

    print("Generation ", generation, " | Mean rewards: ", np.mean(
        rewards), " | Mean of top 5: ", np.mean(top_rewards[:5]))
    # print(rewards)
    print("Top ", top_limit, " scores", sorted_parent_indexes)
    print("Rewards for top: ", top_rewards)

    # setup an empty list for containing children agents
    children_agents = return_children(
        agents, sorted_parent_indexes)

    # kill all agents, and replace them with their children
    agents = children_agents
