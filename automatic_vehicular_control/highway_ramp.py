from automatic_vehicular_control.exp import *
from automatic_vehicular_control.env import *
from automatic_vehicular_control.u import *

class RampEnv(Env):
    # https://flow-project.github.io/papers/08569485.pdf
    def def_sumo(self):
        c = self.c
        builder = NetBuilder()
        nodes = builder.add_nodes(Namespace(x=x, y=y) for x, y in [
            (0, 0),
            (c.premerge_distance, 0),
            (c.premerge_distance + c.merge_distance, 0),
            (c.premerge_distance + c.merge_distance + c.postmerge_distance, 0),
            (c.premerge_distance - 100 * np.cos(np.pi / 4), -100 * np.sin(np.pi / 4))
        ])
        builder.chain(nodes[[0, 1, 2, 3]], edge_attrs=[
            {}, {'numLanes': 2}, {}
        ], lane_maps=[
            {0: 1}, {0: 0, 1: 0}
        ], route_id='highway')
        builder.chain(nodes[[4, 1, 2, 3]], edge_attrs=[
            {}, {'numLanes': 2}, {}
        ], lane_maps=[
            {0: 0}, {0: 0, 1: 0}
        ], route_id='ramp')
        nodes, edges, connections, routes = builder.build()
        nodes[2].type = 'zipper'

        routes = E('routes',
            *routes,
            E('flow', **FLOW(f'f_highway', type='generic', route='highway', departSpeed=c.highway_depart_speed, vehsPerHour=c.highway_flow_rate)),
            E('flow', **FLOW(f'f_ramp', type='human', route='ramp', departSpeed=c.ramp_depart_speed, vehsPerHour=c.ramp_flow_rate))
        )
        idm = {**IDM, **dict(accel=1, decel=1.5, minGap=2)}
        additional = E('additional',
            E('vType', id='generic', **idm),
            E('vType', id='rl', **idm),
            E('vType', id='human', **idm),
        )
        sumo_args = {'collision.action': COLLISION.remove}
        return super().def_sumo(nodes, edges, connections, routes, additional, sumo_args=sumo_args)

    def step(self, action=[]):
        c = self.c
        ts = self.ts
        max_dist = (c.premerge_distance + c.merge_distance) if c.global_obs else 100
        max_speed = c.max_speed
        human_type = ts.types.human
        rl_type = ts.types.rl

        prev_rls = sorted(rl_type.vehicles, key=lambda x: x.id)
        for rl, act in zip(prev_rls, action):
            if c.handcraft:
                route, edge, lane = rl.route, rl.edge, rl.lane
                leader, dist = rl.leader()
                level = 1
                if edge.id == 'e_n_0.0_n_400.0':
                    if rl.laneposition < 100:
                        leaders = list(rl.leaders())
                        if len(leaders) > 20:
                            level = 0
                        else:
                            level = (0.75 * np.sign(c.handcraft - rl.speed) + 1) / 2
                        ts.accel(rl, (level * 2 - 1) * (c.max_accel if level > 0.5 else c.max_decel))
                continue
            if not isinstance(act, (int, np.integer)):
                act = (act - c.low) / (1 - c.low)
            if c.act_type.startswith('accel'):
                level = act[0] if c.act_type == 'accel' else act / (c.n_actions - 1)
                ts.accel(rl, (level * 2 - 1) * (c.max_accel if level > 0.5 else c.max_decel))
            else:
                if c.act_type == 'continuous':
                    level = act[0]
                elif c.act_type == 'discretize':
                    level = min(int(act[0] * c.n_actions), c.n_actions - 1) / (c.n_actions - 1)
                elif c.act_type == 'discrete':
                    level = act / (c.n_actions - 1)
                ts.set_max_speed(rl, max_speed * level)

        super().step()

        route = ts.routes.highway
        obs, ids = [], []
        for veh in sorted(rl_type.vehicles, key=lambda v: v.id):
            speed, edge, lane = veh.speed, veh.edge, veh.lane
            merge_dist = max_dist

            lead_speed = follow_speed = other_speed = 0
            other_follow_dist = other_merge_dist = lead_dist = follow_dist = max_dist

            leader, dist = veh.leader()
            if leader: lead_speed, lead_dist = leader.speed, dist

            follower, dist = veh.follower()
            if follower: follow_speed, follow_dist = follower.speed, dist

            if c.global_obs:
                jun_edge = edge.next(route)
                while jun_edge and not (len(jun_edge.lanes) == 2 and jun_edge.lanes[0].get('junction')):
                    jun_edge = jun_edge.next(route)
                if jun_edge:
                    merge_dist = lane.length - veh.laneposition
                    next_edge = edge.next(route)
                    while next_edge is not jun_edge:
                        merge_dist += next_edge.length
                        next_edge = next_edge.next(route)

                    other_lane = jun_edge.lanes[0]
                    for other_veh, other_merge_dist in other_lane.prev_vehicles(0, route=ts.routes.ramp):
                        other_speed = other_veh.speed
                        break
                obs.append([merge_dist, speed, lead_dist, lead_speed, follow_dist, follow_speed, other_merge_dist, other_speed])
            else:
                next_lane = lane.next(route)
                if next_lane and next_lane.get('junction'):
                    if len(edge.lanes) == 2:
                        other_lane = edge.lanes[0]
                        pos = veh.laneposition
                        for other_veh, other_follow_dist in other_lane.prev_vehicles(pos, route=ts.routes.ramp):
                            other_speed = other_veh.speed
                            break
                obs.append([speed, lead_dist, lead_speed, follow_dist, follow_speed, other_follow_dist, other_speed])
            ids.append(veh.id)
        obs = np.array(obs).reshape(-1, c._n_obs) / ([*lif(c.global_obs, max_dist), max_speed] + [max_dist, max_speed] * 3)
        obs = np.clip(obs, 0, 1).astype(np.float32) * (1 - c.low) + c.low
        reward = len(ts.new_arrived) - c.collision_coef * len(ts.new_collided)
        return Namespace(obs=obs, id=ids, reward=reward)

class Ramp(Main):
    def create_env(c):
        return RampEnv(c)

    @property
    def observation_space(c):
        low = np.full(c._n_obs, c.low)
        return Box(low, np.ones_like(low))

    @property
    def action_space(c):
        assert c.act_type in ['discretize', 'discrete', 'continuous', 'accel', 'accel_discrete']
        if c.act_type in ['discretize', 'continuous', 'accel']:
            return Box(low=c.low, high=1, shape=(1,), dtype=np.float32)
        elif c.act_type in ['discrete', 'accel_discrete']:
            return Discrete(c.n_actions)

    def on_rollout_end(c, rollout, stats, ii=None, n_ii=None):
        log = c.get_log_ii(ii, n_ii)
        step_obs_ = rollout.obs
        step_obs = step_obs_[:-1]

        ret, _ = calc_adv(rollout.reward, c.gamma)

        n_veh = np.array([len(o) for o in step_obs])
        step_ret = [[r] * nv for r, nv in zip(ret, n_veh)]
        rollout.update(obs=step_obs, ret=step_ret)

        step_id_ = rollout.pop('id')
        id = np.concatenate(step_id_[:-1])
        id_unique = np.unique(id)

        reward = np.array(rollout.pop('reward'))

        log(**stats)
        log(reward_mean=reward.mean(), reward_sum=reward.sum())
        log(n_veh_step_mean=n_veh.mean(), n_veh_step_sum=n_veh.sum(), n_veh_unique=len(id_unique))
        return rollout

if __name__ == '__main__':
    c = Ramp.from_args(globals(), locals()).setdefaults(
        warmup_steps=100,
        horizon=2000,
        n_steps=100,
        step_save=5,

        premerge_distance=400,
        merge_distance=100,
        postmerge_distance=30,
        av_frac=0.1,
        sim_step=0.5,
        max_speed=30,
        highway_depart_speed=10,
        ramp_depart_speed=0,
        highway_flow_rate=2000,
        ramp_flow_rate=300,
        global_obs=False,
        handcraft=False,

        generic_type='default',
        speed_mode=SPEED_MODE.all_checks,
        collision_coef=5, # If there's a collision, it always involves an even number of vehicles

        act_type='accel_discrete',
        max_accel=1,
        max_decel=1.5,
        n_actions=3,
        low=-1,

        render=False,

        alg=PG,
        lr=1e-3,

        gamma=0.99,
        adv_norm=False,
        batch_concat=True,

    )
    c._n_obs = c.global_obs + 1 + 2 + 2 + 2
    c.run()