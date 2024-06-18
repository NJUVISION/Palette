# coding=utf-8
import os
import logging
import numpy as np
import multiprocess as mp
import tensorflow as tf
__all__ = [tf]

# Add your simulation enviroment here to train your own Palette
import your_env as env
import a3c_agent as agent
import load_trace
import sys

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


S_INFO = 7  #
S_LEN = 6  # take how many chunks in the past
A_DIM = 7
ACTOR_LR_RATE = 0.00025
CRITIC_LR_RATE = 0.0015
NUM_AGENTS = 4
TRAIN_SEQ_LEN = 600  # take as a train batch
MODEL_SAVE_INTERVAL = 100

M_IN_K = 1000.0
MILLISECONDS_IN_SECOND = 1000.0
FEEDBACK_DURATION = 1000.0  #in milisec
RANDOM_SEED = 42
RAND_RANGE = 10000
SUMMARY_DIR = './results'
LOG_FILE = SUMMARY_DIR + '/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './traces/'

NN_MODEL = None

QUALITY_FACTOR          = 10
POSITIVE_SMOOTH_PENALTY = 0
NEGATIVE_SMOOTH_PENALTY = 0
DELAY_PENALTY           = 0.12 #
EMPTY_PENALTY           = 70

ACTION = [+8, +4, +2, 0, -1, -2, -4]

MIN_CRF = 20
MAX_CRF = 42
CRF_INTERVAL = 4
DEFAULT_ACTION = 3
DEFAULT_CRF = 35

VIDEO_CHUNK_NUM = 6


def compute_reward(crf, last_crf, buffer_size, buffer_empty, avg_frame_delay, packet_loss_rate):

    if crf <= last_crf:
        reward = (MAX_CRF - crf) * QUALITY_FACTOR \
                 - buffer_empty * EMPTY_PENALTY \
                 - avg_frame_delay * DELAY_PENALTY \
                 - abs(last_crf - crf) * POSITIVE_SMOOTH_PENALTY
    else:
        reward = (MAX_CRF - crf) * QUALITY_FACTOR \
                 - buffer_empty * EMPTY_PENALTY \
                 - avg_frame_delay * DELAY_PENALTY \
                 - abs(crf - last_crf) * NEGATIVE_SMOOTH_PENALTY \
    
    return reward

def testing(episode, nn_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    os.system('mkdir ' + TEST_LOG_FOLDER)
    
    # run test script
    os.system('python rl_test.py ' + nn_model)
    
    # append test performance to the log
    rewards = []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward = []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.sum(reward[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write((str(episode) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n').encode())
    log_file.flush()


def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='a',
                        level=logging.INFO)

    with tf.Session(config=config) as sess, open(LOG_FILE + '_test', 'wb') as test_log_file:

        actor = agent.ActorNetwork(sess,
                                   state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                   learning_rate=ACTOR_LR_RATE)
        critic = agent.CriticNetwork(sess,
                                     state_dim=[S_INFO, S_LEN],
                                     learning_rate=CRITIC_LR_RATE)


        summary_ops, summary_vars = agent.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        episode = 0
        nn_model = NN_MODEL
        if nn_model is not None:
            saver.restore(sess, nn_model)
            print("Model restored.")
            parse = NN_MODEL[10:-5].split('_')
            episode = int(parse[-1])

        maxreward = -1000
        # assemble experiences from agents, compute the gradients
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0
            min_reward = 0.0
            
            actor_gradient_batch = []
            critic_gradient_batch = []

            nreward = TRAIN_SEQ_LEN
            for i in range(NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()


                actor_gradient, critic_gradient, td_batch = \
                    agent.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)


                nreward = min(nreward,len(actor_gradient))

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                if(np.mean(r_batch) < min_reward):
                    min_reward = np.mean(r_batch)
                total_reward += np.mean(r_batch)
                total_td_loss += np.mean(td_batch)
                # total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.mean(info['entropy'])


            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            actor_gradient_batch = np.divide(actor_gradient_batch , NUM_AGENTS)
            critic_gradient_batch = np.divide(critic_gradient_batch , NUM_AGENTS)
            for j in range(1,nreward):
                for i in range(NUM_AGENTS):
                    actor_gradient_batch[0][j] = np.add(actor_gradient_batch[0][j] , actor_gradient_batch[i][j])
                    critic_gradient_batch[0][j] = np.add(critic_gradient_batch[0][j] , critic_gradient_batch[i][j])

            mean_actor_gradient_batch = actor_gradient_batch[0]
            mean_critic_gradient_batch = critic_gradient_batch[0]
                
            actor.apply_gradients(mean_actor_gradient_batch)
            critic.apply_gradients(mean_critic_gradient_batch)

            # log training information
            episode += 1
            avg_reward = total_reward  / total_agents
            avg_td_loss = total_td_loss / total_agents
            avg_entropy = total_entropy / total_agents

            logging.info("episode:%06d\tTD_loss:%6.5f\tAvg_reward:%8.2f\tMin_reward:%8.2f\tAvg_entropy:%7.6f"%\
                         (episode,avg_td_loss,avg_reward,min_reward,avg_entropy))
            print("episode:%06d\tTD_loss:%6.5f\tAvg_reward:%8.2f\tMin_reward:%8.2f\tAvg_entropy:%7.6f"%\
                         (episode,avg_td_loss,avg_reward,min_reward,avg_entropy))

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: min_reward,
                summary_vars[3]: avg_entropy
            })

            writer.add_summary(summary_str, episode)
            writer.flush()

            if episode % MODEL_SAVE_INTERVAL == 0:
                print ("---------episode %d--------" % episode)
                if(episode % 10000 == 0):
                    maxreward = 0
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(episode) + ".ckpt")
                logging.info("Model saved in file: " + save_path)
                # testing(episode,
                #     SUMMARY_DIR + "/nn_model_ep_" + str(episode) + ".ckpt",
                #     test_log_file)

                if(avg_reward >= maxreward ):
                    maxreward = avg_reward
                    os.system('cp ' + SUMMARY_DIR + "/nn_model_ep_" + str(episode) + ".ckpt.* ./testmodel/")

            if episode == 20000:
                sys.exit(0)

def work_agent(agent_id, all_cooked_time, all_cooked_bw, all_file_names, net_params_queue, exp_queue):

    # create the simulation environment for the agent
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              all_file_names=all_file_names,
                              random_seed=agent_id)

    with tf.Session(config=config) as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'wb') as log_file:
        actor = agent.ActorNetwork(sess,
                                   state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                   learning_rate=ACTOR_LR_RATE)
        critic = agent.CriticNetwork(sess,
                                     state_dim=[S_INFO, S_LEN],
                                     learning_rate=CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        action = DEFAULT_ACTION
        crf = DEFAULT_CRF
        last_crf = DEFAULT_CRF

        action_vec = np.zeros(A_DIM)
        action_vec[action] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0
        rand_times = 1
        while True:

            # interect with the environment and get observations
            buffer_size, packet_loss_rate, \
            avg_rcv_bitrate, nack_sent_count, end_of_video, \
            iframe_flag, rebuffer, frame_delay, avg_frame_delay, vcc, frame_list_len, \
            preframedec, iid, vmaf, SI, TI = net_env.get_video_chunk(crf)

            time_stamp += 1 # in frame

            reward = compute_reward(crf,last_crf,buffer_size,rebuffer,avg_frame_delay,packet_loss_rate)
            r_batch.append(reward)

            log_file.write(("%09d\t%2d\t%d\t%4.1f\t%4.1f\t%-03.1f\t%2.1f\t%6.2f\t%5.3f\t%.2f\t%4.3f\t%8.1f\t"
                           %(time_stamp,crf,iframe_flag,SI,TI,buffer_size,rebuffer,avg_rcv_bitrate,
                             frame_delay,avg_frame_delay,packet_loss_rate,reward)).encode())
            log_file.flush()

            last_crf = crf

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # prepare states
            state[0, -1] = rebuffer / VIDEO_CHUNK_NUM
            state[1, -1] = (5 + avg_frame_delay) / 500  # rtt/2 in ms
            state[2, -1] = packet_loss_rate
            state[3, -1] = iframe_flag
            state[4, -1] = (crf - MIN_CRF) / (MAX_CRF - MIN_CRF)   #last quality
            state[5, -1] = SI / 20.0
            state[6, -1] = TI / 10.0

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))

            # if (agent_id == 0): print(" ".join(str(format(i, '.3f')) for i in action_prob[0]))
            log_file.write((" ".join(str(format(i, '.3f')) for i in action_prob[0]) + '\n').encode())
            log_file.flush()

            action_cumsum = np.cumsum(action_prob)
            hitcount = np.zeros(A_DIM)
            for i in range(rand_times):
                hit = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                hitcount[hit] = hitcount[hit] + 1
            action = hitcount.argmax()

            # inference
            # action = action_prob.argmax()

            crf = last_crf + ACTION[int(action)]
            crf = max(min(crf, MAX_CRF), MIN_CRF)

            entropy_record.append(agent.compute_entropy(action_prob[0]))

            # report experience to the coordinator
            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first step
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)
                # _ = actor.set_entropy_weight()
                # print(_)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                log_file.write(('\n').encode())  # so that in the log we know where video ends

            # store the state and action into batches
            if end_of_video:
                last_crf = DEFAULT_CRF
                crf = DEFAULT_CRF  # use the default action here
                action = DEFAULT_ACTION
 
                action_vec = np.zeros(A_DIM)
                action_vec[action] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
            else:
                s_batch.append(state)

                action_vec = np.zeros(A_DIM)
                action_vec[action] = 1
                a_batch.append(action_vec)


def main():

    np.random.seed(RANDOM_SEED)
    #assert len(MODIFY_BIT_RATE) == A_DIM

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRAIN_TRACES)
    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=work_agent,
                                 args=(i, all_cooked_time, all_cooked_bw, all_file_names,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()
    os.system('chmod -R 777 ' + SUMMARY_DIR)
    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
