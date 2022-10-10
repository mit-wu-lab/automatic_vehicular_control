from automatic_vehicular_control.exp import *
from automatic_vehicular_control.env import *
from automatic_vehicular_control.u import *

class RingEnv(Env):
    def def_sumo(self):
        c = self.c
        r = c.circumference / (2 * np.pi)
        nodes = E('nodes',
            E('node', id='bottom', x=0, y=-r),
            E('node', id='top', x=0, y=r),
        )

        get_shape = lambda start_angle, end_angle: ' '.join('%.5f,%.5f' % (r * np.cos(i), r * np.sin(i)) for i in np.linspace(start_angle, end_angle, 80))
        edges = E('edges',
            E('edge', **{'id': 'right', 'from': 'bottom', 'to': 'top', 'length': c.circumference / 2, 'shape': get_shape(-np.pi / 2, np.pi / 2), 'numLanes': c.n_lanes}),
            E('edge', **{'id': 'left', 'from': 'top', 'to': 'bottom', 'length': c.circumference / 2, 'shape': get_shape(np.pi / 2, np.pi * 3 / 2), 'numLanes': c.n_lanes}),
        )

        connections = E('connections',
            *[E('connection', **{'from': 'left', 'to': 'right', 'fromLane': i, 'toLane': i}) for i in range(c.n_lanes)],
            *[E('connection', **{'from': 'right', 'to': 'left', 'fromLane': i, 'toLane': i}) for i in range(c.n_lanes)],
        )

        additional = E('additional',
            E('vType', id='human', **{**IDM, **LC2013, **dict(accel=1, decel=1.5, minGap=2, sigma=c.sigma)}),
            E('vType', id='rl', **{**IDM, **LC2013, **dict(accel=1, decel=1.5, minGap=2, sigma=0)}),
            *build_closed_route(edges, c.n_veh, c.av, space=c.initial_space)
        )
        return super().def_sumo(nodes, edges, connections, additional)

    def reset_sumo(self):
        c = self.c
        if c.circumference_range:
            c.circumference = np.random.randint(*c.circumference_range)
        return super().reset_sumo()

    @property
    def stats(self):
        c = self.c
        stats = {k: v for k, v in super().stats.items() if 'flow' not in k}
        stats['circumference'] = c.circumference
        return stats

    def step(self, action=None):
        c = self.c
        ts = self.ts
        max_speed = c.max_speed
        circ_max = max_dist = c.circumference_max
        circ_min = c.circumference_min
        rl_type = ts.types.rl

        if not rl_type.vehicles:
            super().step()
            return c.observation_space.low, 0, False, 0

        rl = nexti(rl_type.vehicles)
        if action is not None: # action is None only right after reset
            ts.tc.vehicle.setMinGap(rl.id, 0) # Set the minGap to 0 after the warmup period so the vehicle doesn't crash during warmup
            accel, lc = (action, None) if not c.lc_av else action if c.lc_act_type == 'continuous' else (action['accel'], action['lc'])
            if isinstance(accel, np.ndarray): accel = accel.item()
            if isinstance(lc, np.ndarray): lc = lc.item()
            if c.norm_action and isinstance(accel, (float, np.floating)):
                accel = (accel - c.low) / (c.high - c.low)
            if c.norm_action and isinstance(lc, (float, np.floating)):
                lc = bool(np.round((lc - c.low) / (c.high - c.low)))

            if c.get('handcraft'):
                accel = (0.75 * np.sign(c.handcraft - rl.speed) + 1) / 2
                lc = True
                if c.get('handcraft_lc'):
                    if c.handcraft_lc == 'off':
                        lc = False
                    elif c.handcraft_lc == 'stabilize':
                        other_lane = rl.lane.left or rl.lane.right
                        oleader, odist = other_lane.next_vehicle(rl.laneposition, route=rl.route)
                        ofollower, ofdist = other_lane.prev_vehicle(rl.laneposition, route=rl.route)
                        if odist + ofdist < 7 and odist > 3:
                            lc = True
                        else:
                            lc = False

            if c.act_type == 'accel_discrete':
                ts.accel(rl, accel / (c.n_actions - 1))
            elif c.act_type == 'accel':
                if c.norm_action:
                    accel = (level * 2 - 1) * (c.max_accel if accel > 0.5 else c.max_decel)
                ts.accel(rl, accel)
            else:
                if c.act_type == 'continuous':
                    level = accel
                elif c.act_type == 'discretize':
                    level = min(int(accel * c.n_actions), c.n_actions - 1) / (c.n_actions - 1)
                elif c.act_type == 'discrete':
                    level = accel / (c.n_actions - 1)
                ts.set_max_speed(rl, max_speed * level)
            if c.n_lanes > 1:
                if c.symmetric_action if c.symmetric_action is not None else c.symmetric:
                    if lc:
                        ts.lane_change(rl, -1 if rl.lane_index % 2 else +1)
                else:
                    ts.lane_change_to(rl, lc)

        super().step()

        if len(ts.new_arrived | ts.new_collided):
            print('Detected collision')
            return c.observation_space.low, -c.collision_penalty, True, None
        elif len(ts.vehicles) < c.n_veh:
            print('Bad initialization occurred, fix the initialization function')
            return c.observation_space.low, 0, True, None

        leader, dist = rl.leader()
        if c.n_lanes == 1:
            obs = [rl.speed / max_speed, leader.speed / max_speed, dist / max_dist]
            if c.circ_feature:
                obs.append((c.circumference - circ_min) / (circ_max - circ_min))
            if c.accel_feature:
                obs.append(0 if leader.prev_speed is None else (leader.speed - leader.speed) / max_speed)
        elif c.n_lanes == 2:
            lane = rl.lane
            follower, fdist = rl.follower()
            if c.symmetric:
                other_lane = rl.lane.left or rl.lane.right
                oleader, odist = other_lane.next_vehicle(rl.laneposition, route=rl.route)
                ofollower, ofdist = other_lane.prev_vehicle(rl.laneposition, route=rl.route)
                obs = np.concatenate([
                    np.array([rl.speed, leader.speed, oleader.speed, follower.speed, ofollower.speed]) / max_speed,
                    np.array([dist, odist, fdist, ofdist]) / max_dist
                ])
            else:
                obs = [rl.speed]
                for lane in rl.edge.lanes:
                    is_rl_lane = lane == rl.lane
                    if is_rl_lane:
                        obs.extend([is_rl_lane, dist, leader.speed, fdist, follower.speed])
                    else:
                        oleader, odist = lane.next_vehicle(rl.laneposition, route=rl.route)
                        ofollower, ofdist = lane.prev_vehicle(rl.laneposition, route=rl.route)
                        obs.extend([is_rl_lane, odist, oleader.speed, ofdist, ofollower.speed])
                obs = np.array(obs) / [max_speed, *([1, max_dist, max_speed, max_dist, max_speed] * 2)]
        else:
            obs = [rl.speed]
            follower, fdist = rl.follower()
            for lane in rl.edge.lanes:
                is_rl_lane = lane == rl.lane
                if is_rl_lane:
                    obs.extend([is_rl_lane, dist, leader.speed, fdist, follower.speed])
                else:
                    oleader, odist = lane.next_vehicle(rl.laneposition, route=rl.route)
                    ofollower, ofdist = lane.prev_vehicle(rl.laneposition, route=rl.route)
                    obs.extend([is_rl_lane, odist, oleader.speed, ofdist, ofollower.speed])
            obs = np.array(obs) / [max_speed, *([1, max_dist, max_speed, max_dist, max_speed] * 3)]
        obs = np.clip(obs, 0, 1) * (1 - c.low) + c.low
        reward = np.mean([v.speed for v in (ts.vehicles if c.global_reward else rl_type.vehicles)])
        if c.accel_penalty and hasattr(self, 'last_speed'):
            reward -= c.accel_penalty * np.abs(rl.speed - self.last_speed) / c.sim_step
        self.last_speed = rl.speed

        return obs.astype(np.float32), reward, False, None

class Ring(Main):
    def create_env(c):
        return NormEnv(c, RingEnv(c))

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(c._n_obs,), dtype=np.float32)

    @property
    def action_space(c):
        c.setdefaults(lc_av=False)
        assert c.act_type in ['discretize', 'discrete', 'continuous', 'accel', 'accel_discrete']
        if c.act_type in ['discretize', 'continuous', 'accel']:
            if not c.lc_av or c.lc_act_type == 'continuous':
                return Box(low=c.low, high=c.high, shape=(1 + bool(c.lc_av),), dtype=np.float32)
            elif c.lc_act_type == 'discrete':
                return Namespace(accel=Box(low=c.low, high=c.high, shape=(1,), dtype=np.float32), lc=Discrete(c.lc_av))
        elif c.act_type in ['discrete', 'accel_discrete']:
            if c.lc_av:
                return Namespace(accel=Discrete(c.n_actions), lc=Discrete(c.lc_av))
            return Discrete(c.n_actions)

    def on_train_start(c):
        super().on_train_start()
        if c.get('last_unbiased'):
            c._model.p_head[-1].bias.data[c.lc_av:] = 0

    def on_step_end(c, gd_stats):
        super().on_step_end(gd_stats)
        if c.get('last_unbiased'):
            c._model.p_head[-1].bias.data[c.lc_av:] = 0

if __name__ == '__main__':
    c = Ring.from_args(globals(), locals()).setdefaults(
        n_lanes=1,
        horizon=3000,
        warmup_steps=1000,
        sim_step=0.1,
        av=1,
        max_speed=10,
        max_accel=0.5,
        max_decel=0.5,
        circumference=250,
        circumference_max=300,
        circumference_min=200,
        circumference_range=None,
        initial_space='free',
        sigma=0.2,

        circ_feature=False,
        accel_feature=False,
        act_type='accel',
        lc_act_type='discrete',
        low=-1,
        high=1,
        norm_action=True,
        global_reward=False,
        accel_penalty=0,
        collision_penalty=100,

        n_steps=100,
        gamma=0.999,
        alg=PG,
        norm_reward=True,
        center_reward=True,
        adv_norm=False,
        step_save=None,

        render=False,
    )
    if c.n_lanes == 1:
        c.setdefaults(n_veh=22, _n_obs=3 + c.circ_feature + c.accel_feature)
    elif c.n_lanes == 2:
        c.setdefaults(n_veh=44, lc_mode=LC_MODE.no_lat_collide, symmetric=False, symmetric_action=None, lc_av=2)
        c._n_obs = (1 + 2 * 2 * 2) if c.symmetric else (1 + 2 * 5)
    elif c.n_lanes == 3:
        c.setdefaults(n_veh=66, lc_mode=LC_MODE.no_lat_collide, symmetric=False, symmetric_action=None, lc_av=3, _n_obs=1 + 3 * (1 + 2 * 2))
    c.step_save = c.step_save or min(5, c.n_steps // 10)
    c.redef_sumo = bool(c.circumference_range)
    c.run()
