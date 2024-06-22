import os
import copy
import numpy as np
from utils import Logger
from agents import get_agent
from buffer import ReplayBuffer
from agents import Agent
import gymnasium as gym

def evaluate(eval_env, start_loc, goal_loc, agent, eval_total_steps, logger:Logger):
    eval_acc_reward = 0
    eval_step = 0
    eval_done = False
    eval_trunacted = False
    eval_obs, eval_info = eval_env.reset(options={'start_loc':start_loc, 'goal_loc':goal_loc})
    list_max_Q = []
    while eval_step < eval_total_steps:
        if eval_done or eval_trunacted:
            eval_obs, eval_info = eval_env.reset(options={'start_loc':start_loc, 'goal_loc':goal_loc})
        eval_action, action_info = agent.choose_action(eval_obs, greedy=True)
        eval_next_obs, eval_reward, eval_done, eval_trunacted, eval_info = eval_env.step(eval_action)
        list_max_Q.append(action_info['Q'])
        eval_step += 1
        # eval_acc_reward += eval_reward
        eval_obs = eval_next_obs
    return {"avg_max_Q": np.mean(list_max_Q)}

def run_policy(env, start_loc, goal_loc, agent, discount_factor):
    """
    Runs the policy derived from the Q-table and computes the actual discounted returns.
    
    Parameters:
    env - The environment (assumes OpenAI Gym-like environment)
    Q - The final Q-table
    discount_factor - The discount factor for future rewards
    max_steps - The maximum number of steps to run the policy

    Returns:
    returns - List of actual discounted returns for each visited state in the trajectory
    """
    state, info = env.reset(options={'start_loc':start_loc, 'goal_loc':goal_loc})
    done = False
    truncated = False
    total_return = 0.0
    steps = 0
    trajectory = []
    
    # Run the policy until the episode is done or max_steps is reached
    while not done and not truncated :
        # Choose the best action from the Q-table
        action, action_info = agent.choose_action(state, greedy=True)
        
        # Take the action in the environment
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Append the current state and reward to the trajectory
        trajectory.append((state, next_state, reward))
        
        # Update state
        state = next_state
        steps += 1
    
    # Calculate the actual discounted returns
    returns = []
    G = 0
    for _, _, reward in reversed(trajectory):
        G = reward + discount_factor * G
        returns.append(G)
    
    # Reverse the returns list to correspond to the original trajectory order
    returns.reverse()
    
    return returns

def run(env, start_loc, goal_loc, agent: Agent, config: dict, repeat_idx: int, total_steps: int, logger: Logger,):
    
    replay_buffer = ReplayBuffer(obs_dim=1, size=config['buffer_size'], batch_size=config['batch_size'])
            
    obs, info = env.reset(options={'start_loc':start_loc, 'goal_loc':goal_loc})
    # done = env.unwrapped.done
    done=False
    truncated = False
    episode_reward = 0
    episode_idx = 0
    learn_info_list = []
    trajectory = [obs]
    n_reset_buffer = 0
    n_learn = 0
    n_on_policy = 0
    running_avg = 1.0
    for step in range(total_steps):
        if done or truncated or step == total_steps-1:
            # print(f'repeat_idx: {repeat_idx} | step: {step} | episode reward:{episode_reward} | maxQ: {np.max(agent.Q)} | eps: {agent.eps}')
            data = {"repeat_idx": repeat_idx, "episode_idx": episode_idx, "step": step, "reward": episode_reward, "maxQ": np.max(agent.Q), "eps": agent.eps, "len_episode": len(trajectory)}
            data.update(info)
            #update done and truncated
            data.update({"done": done, "truncated": truncated})

            if done or truncated:
                if env.unwrapped.agent_xy == env.unwrapped.goals_xy[0]:
                    data.update({"reached_goal": 0})
                else:
                    data.update({"reached_goal": 1})
            logger.dump_episodic_data(data)

            obs, info = env.reset(options={'start_loc':start_loc, 'goal_loc':goal_loc})
            episode_reward = 0
            episode_idx += 1
                
            # print(f"repeat:{repeat_idx} trajectory: {trajectory} \n")
            # with open("results/temp.log", 'a') as f:
            #     f.write(f"repeat:{repeat_idx} step {step} done {done} truncated {truncated}trajectory: {trajectory}" + '\n')
            trajectory = [obs]
                
        if (step+1) % 100 == 0:    
            logger.dump({"repeat_idx": repeat_idx,"step": step})
                
        action, action_info = agent.choose_action(obs)
        agent.decay_explore()
        new_obs, reward, done, truncated, info = env.step(action)
        replay_buffer.store(obs, action, reward, new_obs, float(done), action_info['action_prob'])
        # logger.log(key='max_Q(s,a)', value=action_info['max_Q'])
        if len(replay_buffer) >= replay_buffer.batch_size:
            if config['algo_name'] == 'VQ-learning' and config["importance_sampling"]>0:
                trans = replay_buffer.sample_batch()
                if config['syncVQ']:
                    learn_info = agent.learn_syncVQ(trans['obs'], trans['acts'], trans['rews'], trans['next_obs'], trans['done'], prob_action=trans['prob_acts'])
                else:
                    learn_info = agent.learn(trans['obs'], trans['acts'], trans['rews'], trans['next_obs'], trans['done'], prob_action=trans['prob_acts'])
                n_learn += 1
                n_on_policy += 1
                # running_avg = (running_avg + np.mean(learn_info['mask_'])) / (n_on_policy + 1)
                # running_avg = running_avg * 0.5 + 0.5 * np.mean(learn_info['mask_'])
                # running_avg = np.mean(buf)
                running_avg = np.mean(learn_info['mask_'])
                if running_avg < config["importance_sampling"]:
                    replay_buffer.reset()
                    n_reset_buffer += 1
                    if config['syncVQ']:
                        agent.sync_VQ()
                    # running_avg = 0
                    # n_on_policy = 0
                else: 
                    pass
                    # print(f"np.mean(l_on_policy): {np.mean(l_on_policy)}")
                    
                learn_info.update(
                    {
                        "n_learn": n_learn, 
                        "n_reset_buffer": n_reset_buffer, 
                        "n_on_policy": n_on_policy,
                        "running_avg": running_avg,
                    }
                    )
                learn_info_list.append(learn_info)
                for k, v in learn_info.items():
                    logger.log(key=k, value=v)
            else:
                trans = replay_buffer.sample_batch()
                learn_info = agent.learn(trans['obs'], trans['acts'], trans['rews'], trans['next_obs'], trans['done'], prob_action=trans['prob_acts'])
                learn_info_list.append(learn_info)
                for k, v in learn_info.items():
                    logger.log(key=k, value=v)
            if config['noisy_update'] > 0.0:
                agent.simulate_noisy_update()
        trajectory.append(new_obs)

        # [Evaluation]
        if step % config['eval_freq'] == 0:
            eval_dict = evaluate(eval_env=copy.deepcopy(env), start_loc=start_loc, goal_loc=goal_loc, agent=agent, eval_total_steps=config['eval_total_steps'], logger=logger)
            eval_dict.update({"step": step, "repeat_idx": repeat_idx})
            logger.dump_data(eval_dict, path=logger.eval_log_path) 

        episode_reward += reward
        obs = new_obs

    # print(f"n_reset_buffer: {n_reset_buffer}, n_learn: {n_learn}")    
    env.close()           




def run_experiment(algo_name, n_repeat=8, total_steps=300000):
    MOVES = {
        0: (-1, 0), #UP
        1: (1, 0),  #DOWN
        2: (0, -1), #LEFT
        3: (0, 1),   #RIGHT
        4: (0, 0),   #STAY
        5: (0, 0),
        6: (0, 0),
        7: (0, 0),
        8: (0, 0),
        9: (0, 0),
    }

    # [Load a custom map]
    # obstacle_map = [
    #     "0000000000000",
    #     "0000000000000",
    #     "0000000000000",
    #     "1111000001111",
    #     "0000000000000",
    #     "0000000000000",
    #     "0000000000000",
    # ]
    # start_loc = (6,1)
    # goal_loc = [(0,10), (1,2), (6,10)]
    # map_name = "2roomsBig_rwd0.1(-0.1,1)"


    obstacle_map = [
            "0000000",
            "0000000",
            "0000000",
            "1110111",
            "0000000",
            "0000000",
            "0000000",
        ]

    start_loc = (3,3)
    # goal_loc = [(0,3), (6,3), (1,1)]
    goal_loc = [(0,3), (6,3)]
    map_name = "2rooms_rwd0.1(-0.1,1)"

    env = gym.make(
        'SimpleGrid-v0', 
        obstacle_map=obstacle_map, 
        # render_mode='human',
        MOVES=MOVES, # if None, it will use the default 4 moves
        render_mode=None,
        max_episode_steps=300, #This is for TimeLimit wrapper
    )

    # obstacle_map = [
    #         "000",
    #         "000",
    #     ]
    # start_loc = (1,1)
    # goal_loc = [(0,0), (1,2)]
    # map_name = "2row"

    # obstacle_map = [
    #         "000",
    #         "000",
    #         "000",
    #     ]
    # start_loc = (1,1)
    # goal_loc = [(0,0), (2,2)]
    # map_name = "square"

    # n_repeat = 8
    # total_steps = 300000
    config = {
        # "algo_name": "Q-learning",
        # "algo_name": "VQ-learning",
        # "algo_name": "DoubleQ-learning",
        # "algo_name": "SARSA",
        # "algo_name": "ExpectedSARSA",
        "algo_name": algo_name,
        "eps_start": 0.5,
        "eps_end": 0.0,
        "eps_decay": 0.99997,
        # "temperature_start": 1.0, # for softmax explore, if set to not None, it will be used, overwriting eps
        # "temperature_end": 0.01,
        # "temperature_decay": 0.99995,
        "alpha": 0.1,
        "gamma": 0.99,
        "random_value_init": True,
        "alpha_v": 0.1,
        "alpha_q": 0.1,
        "buffer_size": 1,
        "batch_size": 1,
        "importance_sampling": 0.0, # this will work as a threshold
        "syncVQ": False,
        "eval_freq": 500,
        "eval_total_steps": 200,
        "noisy_update": 0.0,
    }
    algo_name = config["algo_name"]
    if config.get("temperature_start", None) is not None:
        explore_start, explore_end, explore_decay = config["temperature_start"], config["temperature_end"], config["temperature_decay"]
    else:
        explore_start, explore_end, explore_decay = config["eps_start"], config["eps_end"], config["eps_decay"]
    log_dir = f"results/{map_name}/{algo_name}/randinit{int(config['random_value_init'])}_explore[{explore_start},{explore_end},{explore_decay}]_buffer{config['buffer_size']}_{config['batch_size']}_is{config['importance_sampling']}_noisyV{config['noisy_update']}_repeat{n_repeat}_steps{total_steps}" + "--TrueQ-redundantA#2"

    log_paths = {
        "avg_meter": os.path.join(log_dir, "avg_meter.log"),
        "episodic": os.path.join(log_dir, "episodic.log"),
        "true_return": os.path.join(log_dir, "true_return.log"),
        "eval": os.path.join(log_dir, "eval.log"),
    }
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    for log_path in log_paths.values():
        if os.path.exists(log_path):
            os.remove(log_path)

    logger = Logger(log_paths=log_paths)

    # algo = "VQ-learning"
    # if results/temp.log exists, remove it
    if os.path.exists("results/temp.log"):
        os.remove("results/temp.log")

    best_policy = None
    avg_true_return = -np.inf 
    for i_repeat in range(n_repeat):
        agent = get_agent(config, env)
        run(env, start_loc, goal_loc, agent, config=config, repeat_idx=i_repeat, total_steps=total_steps, logger=logger)
        # print(f"repeat {i_repeat} done")

        # [Evaluate the final policy, log and save the new best policy]
        avgs_true_returns = []
        list_true_returns = []
        n_final_evals = 3
        for _ in range(n_final_evals): # considering the transition function of the mdp can be stochastic
            true_returns = run_policy(copy.deepcopy(env), start_loc, goal_loc, agent, discount_factor=config["gamma"])
            list_true_returns.append(true_returns)
            avgs_true_returns.append(np.mean(true_returns))
        if np.mean(avgs_true_returns) > avg_true_return:
            avg_true_return = np.mean(avgs_true_returns)
            best_policy = copy.deepcopy(agent.Q)
            # save the best policy
            np.save(os.path.join(log_dir, "best_policy.npy"), best_policy)

            with open(log_paths["true_return"], 'w') as f:
                pass
            for true_returns in list_true_returns:
                logger.dump_data(true_returns, log_paths["true_return"], overwrite=False)
    
    return log_dir
                    
