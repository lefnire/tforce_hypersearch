import tensorflow as tf
import pdb, json, random, argparse, math
from pprint import pprint
import numpy as np
import pandas as pd
from sqlalchemy.sql import text
from data import conn
from tensorforce.environments import Environment
from tensorforce import Configuration
from tensorforce.agents import agents as agents_dict
from btc_env.btc_env import BitcoinEnvTforce
from tensorforce.execution import Runner, ThreadedRunner
from tensorforce.execution.threaded_runner import WorkerAgent
from tensorforce.contrib.openai_gym import OpenAIGym

"""
Each hyper is specified as `key: {type, vals, requires, hook}`. 
- type: (int|bounded|bool). bool is True|False param, bounded is a float between min & max, int is "choose one" 
    eg 'activation' one of (tanh|elu|selu|..)`)
- vals: the vals this hyper can take on. If type(vals) is primitive, hard-coded at this value. If type is list, then
    (a) min/max specified inside (for bounded); (b) all possible options (for 'int'). If type is dict, then the keys
    are used in the searching (eg, look at the network hyper) and the values are used as the configuration.
- requires: can specify that a hyper should be deleted if some other hyper (a) doesn't exist (type(requires)==str), 
    (b) doesn't equal some value (type(requires)==dict)
- hook: transform this hyper before plugging it into Configuration(). Eg, we'd use type='bounded' for batch size since
    we want to range from min to max (insteaad of listing all possible values); but we'd cast it to an int inside
    hook before using it. (Actually we clamp it to blocks of 8, as you'll see)      
    
The special sauce is specifying hypers as dot-separated keys, like `memory.type`. This allows us to easily 
mix-and-match even within a config-dict. Eg, you can try different combos of hypers within `memory{}` w/o having to 
specify the whole block combo (`memory=({this1,that1}, {this2,that2})`). To use this properly, make sure to specify
a `requires` field where necessary. 
"""

hypers = {}
hypers['agent'] = {
    'exploration': {
        'type': 'int',
        'vals': {
            "type": "epsilon_decay",
            "epsilon": 1.0,
            "epsilon_final": 0.1,
            "epsilon_timesteps": 1e9
        }
    },  # TODO epsilon_anneal
    # TODO preprocessing, batch_observe, reward_preprocessing[dict(type='clip', min=-1, max=1)]
}
hypers['memory_agent'] = {
    'batch_size': {
        'type': 'bounded',
        'vals': [8, 256],
        'hook': lambda x: int(x // 8) * 8
    },
    'memory.type': {
        'type': 'int',
        'vals': ['replay', 'naive-prioritized-replay']
    },
    'memory.random_sampling': {
        'type': 'bool',
        'requires': {'memory.type': 'replay'},
    },
    'memory.capacity': {
        'type': 'bounded',
        'vals': [10000, 100000],  # ensure > batch_size
        'hook': int
    },
    'first_update': {
        'type': 'bounded',
        'vals': [1000, 10000],
        'hook': int
    },
    'update_frequency': {
        'type': 'bounded',
        'vals': [4, 20],
        'hook': int
    },
    'repeat_update': {
        'type': 'bounded',
        'vals': [1, 4],
        'hook': int
    }
}
hypers['batch_agent'] = {
    'batch_size': {
        'type': 'bounded',
        'vals': [8, 256],
        'hook': lambda x: int(x // 8) * 8
    },
    'keep_last_timestep': {
        'type': 'bool'
    }
}
hypers['model'] = {
    'optimizer.type': {
        'type': 'int',
        'vals': ['nadam', 'adam'],  # TODO rmsprop
    },
    'optimizer.learning_rate': {
        'type': 'bounded',
        'vals': [1e-7, 1e-2],
    },
    'optimization_steps': {
        'type': 'bounded',
        'vals': [5, 20],
        'hook': int
    },
    'discount': {
        'type': 'bounded',
        'vals': [.95, .99],
    },
    'normalize_rewards': True,  # can hard-code attrs you find are definite winners
    # TODO variable_noise
}
hypers['distribution_model'] = {
    'entropy_regularization': {
        'type': 'bounded',
        'vals': [0., 1,],
    }
    # distributions_spec (gaussian, beta, etc). Pretty sure meant to handle under-the-hood, investigate
}
hypers['pg_model'] = {
    'baseline_mode': 'states',
    'gae_lambda': {
        'requires': 'baseline_mode',
        'type': 'bounded',
        'vals': [.94, .99],
        'hook': lambda x: None if x < .95 else x
    },
    'baseline_optimizer.optimizer.learning_rate': {
        'requires': 'baseline_mode',
        'type': 'bounded',
        'vals': [1e-7, 1e-2]
    },
    'baseline_optimizer.num_steps': {
        'requires': 'baseline_mode',
        'type': 'bounded',
        'vals': [5, 20],
        'hook': int
    },
}
hypers['pg_prob_ration_model'] = {
    # I don't know what values to use besides the defaults, just guessing. Look into
    'likelihood_ratio_clipping': {
        'type': 'bounded',
        'vals': [0., 1.],
    }
}
hypers['q_model'] = {
    'target_sync_frequency': 10000,  # This effects speed the most - make it a high value
    'target_update_weight': {
        'type': 'bounded',
        'vals': [0., 1.],
    },
    'double_q_model': True,
    'huber_loss': {
        'type': 'bounded',
        'vals': [0., 1.],
        'hook': lambda x: None if x < .001 else x
    }
}

hypers['dqn_agent'] = {
    **hypers['agent'],
    **hypers['memory_agent'],
    **hypers['model'],
    **hypers['distribution_model'],
    **hypers['q_model'],
}
hypers['ppo_agent'] = {  # vpg_agent, trpo_agent
    **hypers['agent'],
    **hypers['batch_agent'],
    **hypers['model'],
    **hypers['distribution_model'],
    **hypers['pg_model'],
    **hypers['pg_prob_ration_model']

}
hypers['ppo_agent']['step_optimizer.learning_rate'] = hypers['ppo_agent'].pop('optimizer.learning_rate')
hypers['ppo_agent']['step_optimizer.type'] = hypers['ppo_agent'].pop('optimizer.type')
del hypers['ppo_agent']['exploration']

hypers['custom'] = {
    'network': {
        'type': 'bounded',
        'vals': {
            2: [
                {'type': 'dense', 'size': 128},
                {'type': 'dense', 'size': 128},
                {'type': 'dense', 'size': 64},
                {'type': 'dense', 'size': 32}
            ],
            1: [
                {'type': 'dense', 'size': 64},
                {'type': 'dense', 'size': 64},
                {'type': 'dense', 'size': 32}
            ],
            0: [
                {'type': 'dense', 'size': 32},
                {'type': 'dense', 'size': 32}
            ],
        },
        'hook': lambda x: math.floor(x)
    },
    'activation': {
        'type': 'int',
        'vals': ['tanh', 'elu', 'relu', 'selu'],
    },
    'dropout': {
        'type': 'bounded',
        'vals': [0., .5],
        'hook': lambda x: None if x < .1 else x
    },
    'l2_regularization': {
        'type': 'bounded',
        'vals': [1e-5, 1e-1]
    }
}


class DotDict(object):
    """
    Utility class that lets you get/set attributes with a dot-seperated string key, like `d = a['b.c.d']` or `a['b.c.d'] = 1`
    """
    def __init__(self, data):
        self._data = data

    def __getitem__(self, path):
        v = self._data
        for k in path.split('.'):
            if k not in v:
                return None
            v = v[k]
        return v

    def __setitem__(self, path, val):
        v = self._data
        path = path.split('.')
        for i, k in enumerate(path):
            if i == len(path) - 1:
                v[k] = val
                return
            elif k in v:
                v = v[k]
            else:
                v[k] = {}
                v = v[k]

    def to_dict(self):
        return self._data


class HSearchEnv(Environment):
    """
    This is the "wrapper" environment (the "inner" environment is the one you're testing against, like Cartpole-v0).
    This env's actions are all the hyperparameters (above). The state is nothing (`[1.]`), and a single episode is
    running the inner-env however many episodes (300). The inner's last-few reward avg is outer's one-episode reward.
    That's one run: make inner-env, run 300, avg reward, return that. The next episode will be a new set of
    hyperparameters (actions); run inner-env from scratch using new hypers.
    """
    def __init__(self, agent='ppo_agent'):
        """
        TODO only tested with ppo_agent. There's some code for dqn_agent, but I haven't tested. Nothing else
        is even attempted implemtned
        """
        hypers_ = hypers[agent].copy()
        hypers_.update(hypers['custom'])

        self.agent = agent
        self.hypers = hypers_
        self.hardcoded = {}
        self.actions_ = {}

        for k, v in hypers_.items():
            if type(v) != dict:
                self.hardcoded[k] = v
            elif v['type'] == 'int':
                self.actions_[k] = dict(type='int', shape=(), num_actions=len(v['vals']))
            elif v['type'] == 'bounded':
                # cast to list in case the keys are the min/max (as in network)
                min, max = np.min(list(v['vals'])), np.max(list(v['vals']))
                self.actions_[k] = dict(type='float', shape=(), min_value=min, max_value=max)
            elif v['type'] == 'bool':
                self.actions_[k] = dict(type='bool', shape=())

    def __str__(self):
        return 'HSearchEnv'

    def close(self):
        pass

    @property
    def actions(self):
        return self.actions_

    @property
    def states(self):
        return {'shape': 1, 'type': 'float'}

    def _action2val(self, k, v):
        hyper = self.hypers[k]
        if 'hook' in hyper:
            v = hyper['hook'](v)
        if hyper['type'] == 'int':
            if type(hyper['vals']) == list:
                return hyper['vals'][v]
            # Else it's a dict. Don't map the values till later (keep them as keys in flat)
        return v

    def _key2val(self, k, v):
        hyper = self.hypers[k]
        if type(hyper) == dict and type(hyper.get('vals', None)) == dict:
            return hyper['vals'][v]
        return v

    def reset(self):
        return [1.]

    def execute(self, actions):
        """
        Bit of confusing logic here where I construct a `flat` dict of hypers from the actions - looks like how hypers
        are specified above ('dot.key.str': val). Then from that we hydrate it as a proper config dict (hydrated).
        Keeping `flat` around because I save the run to the database so can be analyzed w/ a decision tree
        (for feature_importances and the like) and that's a good format, rather than a nested dict.
        :param actions: the hyperparamters
        """
        flat = {k: self._action2val(k, v.item()) for k, v in actions.items()}
        flat.update(self.hardcoded)

        # Ensure dependencies (do after above to make sure the randos have "settled")
        for k in list(flat):
            if k in self.hardcoded: continue
            hyper = self.hypers[k]
            if not (type(hyper) is dict and 'requires' in hyper): continue
            req = hyper['requires']
            # Requirement is a string (require the value's not None). TODO handle nested deps.
            if type(req) is str:
                if not flat[req]: del flat[k]
                continue
            # Requirement is a dict of type {key: value_it_must_equal}. TODO handle multiple deps
            dep_k, dep_v = list(req.items())[0]
            if flat[dep_k] != dep_v:
                del flat[k]

        # TODO put this in hard-coded hyper above?
        if self.agent == 'ppo_agent':
            hydrated = DotDict({
                'baseline_mode': 'states',
                'baseline': {'type': 'custom'},
                'baseline_optimizer': {'type': 'multi_step', 'optimizer': {'type': 'nadam'}},
            })

        # change all a.b=c to {a:{b:c}} (note DotDict class above, I hate and would rather use an off-the-shelf)
        for k, v in flat.items():
            if k not in hypers['custom']:
                hydrated[k] = self._key2val(k, v)
        hydrated = hydrated.to_dict()


        extra = {k: self._key2val(k, v) for k, v in flat.items() if k in hypers['custom']}
        net_spec = extra['network']
        for layer in net_spec:
            layer['activation'] = extra['activation']
            layer['l2_regularization'] = extra['l2_regularization']
            # TODO add dropout

        if flat.get('baseline_mode', None) == 'states':
            hydrated['baseline']['network_spec'] = net_spec

        pprint(hydrated)

        hydrated['scope'] = 'hypersearch'

        env = OpenAIGym('CartPole-v0')
        agent = agents_dict[self.agent](
            states_spec=env.states,
            actions_spec=env.actions,
            network_spec=net_spec,
            config=Configuration(**hydrated)
        )

        n_train, n_test = 230, 20
        runner = Runner(agent=agent, environment=env)
        runner.run(episodes=n_train)  # train
        runner.run(episodes=n_test, deterministic=True)  # test
        # You may need to remove runner.py's close() calls so you have access to runner.episode_rewards, see
        # https://github.com/lefnire/tensorforce/commit/976405729abd7510d375d6aa49659f91e2d30a07
        reward = np.mean(runner.episode_rewards[-n_test:])
        print(flat, f"\nReward={reward}\n\n")

        # # I personally save away the results so I can play with them manually w/ scikitlearn & SQL
        # sql = "insert into runs (hypers, reward_avg, rewards, agent) values (:hypers, :reward_avg, :rewards, :agent)"
        # conn.execute(text(sql), hypers=json.dumps(flat), reward_avg=reward, rewards=ep_results['rewards'], agent='ppo_agent')

        runner.agent.close()
        runner.environment.close()

        next_state, terminal = [1.], False
        return next_state, terminal, reward

def setup_runs_table():
    """Not needed, but if you want to save away your runs to analyze manually"""
    conn.execute("""
        create table if not exists runs
        (
            id SERIAL,
            hypers jsonb not null,
            reward_avg double precision not null,
            rewards double precision[] not null,
            flag varchar(16)
        );
    """)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workers', type=int, default=1, help="Number of workers")
    args = parser.parse_args()

    # This begs the question "isn't that turtles all the way down?". Maybe (still testing), but I'm thinking since it's
    # a single state, I'm thinking this can be a simpler set of default hypers. And where the env itself might be
    # incredibly sensitive to pin-point hyper selection, this outer guy can be a bit more crude. We'll see, I'll
    # report back w/ success/failure.
    network_spec = [
        {'type': 'dense', 'size': 64},
        {'type': 'dense', 'size': 64},
    ]
    config = Configuration(
        batch_size=4,
        batched_observe=0,
        discount=0.
    )
    if args.workers == 1:
        env = HSearchEnv()
        agent = agents_dict['ppo_agent'](
            states_spec=env.states,
            actions_spec=env.actions,
            network_spec=network_spec,
            config=config
        )
        runner = Runner(agent=agent, environment=env)
        runner.run()  # forever (the env will cycle internally)
    else:
        main_agent = None
        agents, envs = [], []

        for i in range(args.workers):
            envs.append(HSearchEnv())
            if i == 0:
                # let the first agent create the model, then create agents with a shared model
                main_agent = agent = agents_dict['ppo_agent'](
                    states_spec=envs[0].states,
                    actions_spec=envs[0].actions,
                    network_spec=network_spec,
                    config=config
                )
            else:
                config.default(main_agent.default_config)
                agent = WorkerAgent(
                    states_spec=envs[0].states,
                    actions_spec=envs[0].actions,
                    network_spec=network_spec,
                    config=config,
                    model=main_agent.model
                )
            agents.append(agent)

        def summary_report(x): pass
        threaded_runner = ThreadedRunner(agents, envs)
        threaded_runner.run(
            episodes=-1,  # forever (the env will cycle internally)
            summary_interval=2000,
            summary_report=summary_report
        )


if __name__ == '__main__':
    main()