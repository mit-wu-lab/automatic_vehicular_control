from automatic_vehicular_control.exp import *
from automatic_vehicular_control.env import *
from automatic_vehicular_control.u import *

class FigureEightEnv(Env):
    def def_sumo(self):
        c = self.c

        r = c.radius
        ring_length = r * (3 * np.pi / 2)
        nodes = E('nodes',
            E('node', id='center', x=r, y=r), # center around (r, r) instead of (0, 0) so that SUMO creates all internal edges symmetrically
            E('node', id='right', x=2 * r, y=r),
            E('node', id='top', x=r, y=2 * r),
            E('node', id='left', x=0, y=r),
            E('node', id='bottom', x=r, y=0),
        )
        center, right, top, left, bottom = nodes
        builder = NetBuilder()
        builder.chain([left, bottom, center, top, right, center, left], edge_attrs=[
            dict(length=ring_length, shape=' '.join(f'{r * np.cos(i):.5f},{r * np.sin(i):.5f}' for i in np.linspace(np.pi / 2, 2 * np.pi, 40))),
            {}, {},
            dict(length=ring_length, shape=' '.join(f'{r * (2 - np.cos(i)):.5f},{r * (2 + np.sin(i)):.5f}' for i in np.linspace(0, 3 * np.pi / 2, 40))),
            {}, {},
        ])
        _, edges, connections, _ = builder.build()

        assert c.av == 0 or c.n_veh % c.av == 0
        v_params = {**IDM, **LC2013, **dict(accel=1, decel=1.5, minGap=c.get('min_gap', 2))}
        additional = E('additional',
            E('vType', id='human', **v_params),
            E('vType', id='rl', **v_params),
            *build_closed_route(edges, c.n_veh, space=c.initial_space, type_fn=lambda i: 'rl' if c.av != 0 and i % (c.n_veh // c.av) == 0 else 'human', depart_speed=c.get('depart_speed', 0), offset=c.get('offset', 0), init_length=c.get('init_length'))
        )
        return super().def_sumo(nodes, edges, connections, additional)

    @property
    def stats(self):
        return {k: v for k, v in super().stats.items() if 'flow' not in k}

    def step(self, action=[]):
        c = self.c
        ts = self.ts
        max_speed = c.max_speed

        ids = np.arange(c.n_veh)
        rl_ids = ids[::c.n_veh // c.av] if c.av else [] # ith vehicle is RL

        if len(action):
            rls = [ts.vehicles[f'{id}'] for id in rl_ids]
            if not isinstance(action, int):
                action = (action - c.low) / (1 - c.low)
            for a, rl in zip(action, rls):
                if c.act_type.startswith('accel'):
                    level = a / (c.n_actions - 1) if c.act_type == 'accel_discrete' else a
                    if c.get('handcraft'):
                        level = (0.75 * np.sign(c.handcraft - rl.speed) + 1) / 2
                        n_followers = c.n_veh // c.av - 1
                        veh, dist = rl, 0
                        for _ in range(n_followers):
                            veh, f_dist = veh.follower()
                            dist += f_dist
                        if dist > c.get('follower_gap', 17) * n_followers:
                            level = (0.75 * np.sign(0.5 - rl.speed) + 1) / 2
                    ts.accel(rl, (level * 2 - 1) * (c.max_accel if level > 0.5 else c.max_decel))
                else:
                    if c.act_type == 'continuous':
                        level = a
                    elif c.act_type == 'discretize':
                        level = min(int(a * c.n_actions), c.n_actions - 1) / (c.n_actions - 1)
                    ts.set_max_speed(rl, max_speed * level)

        super().step()

        if ts.new_arrived | ts.new_collided: # Collision
            print(f'Collision between vehicles {[v.id for v in ts.new_arrived | ts.new_collided]} on step {self._step}')
            return dict(reward=np.full(c.av, -10), done=True)
        elif len(ts.vehicles) not in [0, c.n_veh]:
            print(f'Bad number of initial vehicles {len(ts.vehicles)}, probably due to collision')
            return dict(reward=0, done=True)

        rls = [ts.vehicles[f'{id}'] for id in rl_ids]
        vehs = [ts.vehicles[f'{id}'] for id in ids]

        route = nexti(ts.routes)
        max_dist = max(x.route_position[route] + x.length for x in route.edges)

        state = np.array([(v.edge.route_position[route] + v.laneposition, v.speed) for v in vehs])
        obs = np.array([np.roll(state, -i, axis=0) for i in rl_ids]).reshape(c.av, c.n_veh, 2) / [max_dist, max_speed] # (c.av, c.n_veh, 2)
        obs[:, :, 0] = obs[:, :, 0] - 0.5 * (obs[:, 0, 0] >= 0.5).reshape(c.av, 1) # Subtract the position by 0.5 as needed for symmetry
        obs = obs.reshape(c.av, c._n_obs)
        assert c.av == 0 or 0 <= np.abs(obs).max() < 1, f'Observation out of range: {obs}'

        rl_speeds = np.array([v.speed for v in rls])
        reward = np.mean([v.speed for v in vehs] if c.global_reward else rl_speeds)

        if c.accel_penalty and hasattr(self, 'last_speeds'):
            reward -= c.accel_penalty * np.abs(rl_speeds - self.last_speeds) / c.sim_step
        self.last_speeds = rl_speeds

        return dict(obs=obs.astype(np.float32), id=rl_ids, reward=np.full(c.av, reward))

class FigureEight(Main):
    def create_env(c):
        return NormEnv(c, FigureEightEnv(c))

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(c._n_obs,), dtype=np.float32)

    @property
    def action_space(c):
        assert c.act_type in ['discretize', 'continuous', 'accel', 'accel_discrete']
        if c.act_type == 'accel_discrete':
            return Discrete(c.n_actions)
        return Box(low=c.low, high=1, shape=(1,), dtype=np.float32) # Need to have a nonzero number of actions

if __name__ == '__main__':
    c = FigureEight.from_args(globals(), locals()).setdefaults(
        horizon=3000,
        warmup_steps=100,
        sim_step=0.1,
        n_veh=14,
        av=1,
        max_speed=30,
        max_accel=0.5,
        max_decel=0.5,
        radius=30,
        speed_mode=SPEED_MODE.obey_safe_speed,
        initial_space='free',

        act_type='accel_discrete',
        low=-1,
        n_actions=3,
        global_reward=True,
        accel_penalty=0,

        n_steps=100,
        gamma=0.99,
        alg=PG,
        norm_reward=True,
        center_reward=True,
        adv_norm=False,
        batch_concat=True,
        step_save=5,

        render=False,
    )
    c._n_obs = 2 * c.n_veh
    c.run()
