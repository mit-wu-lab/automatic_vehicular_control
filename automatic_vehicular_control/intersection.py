from automatic_vehicular_control.u import *
from automatic_vehicular_control.exp import *
from automatic_vehicular_control.env import *

class Platoon(Entity):
    pass

class IntersectionEnv(Env):
    def def_sumo(self):
        c = self.c

        types = [E('vType', id='human', **IDM, **LC2013), E('vType', id='rl', **IDM, **LC2013), E('vType', id='generic', **IDM, **LC2013)]
        default_flows = lambda flow_id, route_id, flow_rate: [E('flow', **params) for params in [
            FLOW(f'{flow_id}', type='generic', route=route_id, departSpeed=c.depart_speed, vehsPerHour=flow_rate),
        ] if params.get('vehsPerHour')]

        builder = NetBuilder()
        xys = np.array(np.ones((c.n_rows + 2, c.n_cols + 2)).nonzero()).T * c.length
        if c.enter_length:
            xys = xys.reshape(c.n_rows + 2, c.n_cols + 2, 2)
            xys[1:, :, 0] += c.enter_length - c.length
            xys[:, 1:, 1] += c.enter_length - c.length
            xys = xys.reshape(-1, 2)
        if c.short_exit:
            xys = xys.reshape(c.n_rows + 2, c.n_cols + 2, 2)
            xys[-1, :, 0] += c.short_exit - c.length
            xys[:, -1, 1] += c.short_exit - c.length
            xys = xys.reshape(-1, 2)
        nodes = builder.add_nodes(
            [Namespace(x=x, y=y, type='priority') for y, x in xys]
        ).reshape(c.n_rows + 2, c.n_cols + 2)

        tl = c.setdefault('tl', False)
        if tl:
            c.av_frac = 0
            c.pop('av_range', None)
            c.speed_mode = SPEED_MODE.all_checks

        flows = []
        c.setdefaults(flow_rate_h=c.flow_rate, flow_rate_v=c.flow_rate)
        priority = ['left', 'right'] if c.get('priority', 'vertical') == 'horizontal' else ['up', 'down']
        for direction in c.directions:
            chains = nodes if direction in ['left', 'right'] else nodes.T
            chains = chains if direction in ['up', 'right'] else np.fliplr(chains)
            flow_rate = c.flow_rate_h if direction in ['left', 'right'] else c.flow_rate_v

            edge_attrs = dict(priority=int(direction in priority))
            if c.get('set_edge_speed', True):
                edge_attrs['speed'] = c.max_speed

            for i, chain in enumerate(chains[1:-1]):
                route_id, flow_id = f'r_{direction}_{i}', f'f_{direction}_{i}'
                builder.chain(chain, route_id=route_id, edge_attrs=edge_attrs)
                flows.extend(default_flows(flow_id, route_id, flow_rate))

        tls = []
        if tl:
            tl = 1000000 if tl == 'MaxPressure' else tl
            tl_h, tl_v = tl if isinstance(tl, tuple) else (tl, tl)
            tl_offset = c.get('tl_offset', 'auto')
            yellow = c.get('yellow', 0.5)
            if tl_offset == 'auto':
                offsets = c.length * (np.arange(c.n_rows).reshape(-1, 1) + np.arange(c.n_cols).reshape(1, -1)) / 10
            elif tl_offset == 'same':
                offsets = np.zeros(c.n_rows).reshape(-1, 1) + np.zeros(c.n_cols).reshape(1, -1)
            for node, offset in zip(nodes[1:-1, 1:-1].reshape(-1), offsets.reshape(-1)):
                node.type = 'traffic_light'
                phase_multiple = len(c.directions) // 2
                tls.append(E('tlLogic',
                    E('phase', duration=tl_v, state='Gr' * phase_multiple),
                    *lif(yellow, E('phase', duration=yellow, state='yr' * phase_multiple)),
                    E('phase', duration=tl_h, state='rG' * phase_multiple),
                    *lif(yellow, E('phase', duration=yellow, state='ry' * phase_multiple)),
                id=node.id, offset=offset, type='static', programID='1'))

        nodes, edges, connections, routes = builder.build()
        additional = E('additional', *types, *routes, *flows, *tls)
        return super().def_sumo(nodes, edges, connections, additional)

    def build_platoon(self):
        ts = self.ts
        rl_type = ts.types.rl
        for route in ts.routes:
            vehs = []
            route_offset = 0
            for edge in route.edges:
                for veh in edge.vehicles:
                    veh.route_position = route_offset + veh.laneposition
                    vehs.append(veh)
                route_offset += edge.length

            rl_mask = np.array([veh.type is rl_type for veh in vehs])
            if len(rl_mask) > 1 and c.get('merge_consecutive_avs'):
                rl_mask[1:] = rl_mask[1:] & ~rl_mask[:-1]
            rl_idxs, = rl_mask.nonzero()
            split_idxs = 1 + rl_idxs

            prev = None
            for i, vehs_i in enumerate(np.split(vehs, split_idxs)):
                if not len(vehs_i):
                    continue # Last vehicle is RL, so the last split is empty
                platoon = Platoon(id=f'{route}.platoon_{i}', route=route,
                    vehs=vehs_i, head=vehs_i[-1], tail=vehs_i[0], prev=prev
                )
                if prev is not None:
                    prev.next = platoon
                prev = platoon
                for veh in vehs_i:
                    veh.platoon = platoon
            if prev is not None:
                prev.next = None

    def reset(self):
        c = self.c
        if c.e is False:
            if 'length_range' in c:
                min_, max_ = c.length_range
                c.setdefaults(max_length=max_)
                c.length = np.random.randint(min_, max_ + 1)
            if 'av_range' in c:
                min_, max_ = c.av_range
                c.av_frac = np.random.uniform(min_, max_)
        self.mp_tlast = 0
        while not self.reset_sumo():
            pass
        ret = super().init_env()
        return ret

    def step(self, action=[]):
        c = self.c
        ts = self.ts
        max_dist = c.max_dist
        depart_speed = c.depart_speed
        max_speed = c.max_speed

        rl_type = ts.types.rl
        prev_rls = sorted(rl_type.vehicles, key=lambda x: x.id)

        for rl, act in zip(prev_rls, action):
            if c.handcraft or c.handcraft_tl:
                route, lane = rl.route, rl.lane
                junction = lane.next_junction
                level = 1
                if junction is not ts.sentinel_junction:
                    dist = junction.route_position[route] - rl.route_position
                    if c.handcraft:
                        for cross_lane in lane.next_cross_lanes:
                            cross_veh, cross_dist = cross_lane.prev_vehicle(0, route=nexti(cross_lane.from_routes))
                            level = 1 - int(dist < c.handcraft and cross_veh and (cross_veh.type is not rl_type or dist > cross_dist))
                    elif c.handcraft_tl and dist < 15:
                        t_h, t_v = c.handcraft_tl if isinstance(c.handcraft_tl, tuple) else (c.handcraft_tl, c.handcraft_tl)
                        yellow_time = 0
                        rem = self._step % (t_h + yellow_time + t_v + yellow_time)
                        horizontal_go = 0 <= rem < t_h
                        vertical_go = t_h + yellow_time <= rem
                        horizontal_lane = 'left' in route.id or 'right' in route.id
                        human_remain = False
                        for cross_lane in lane.next_cross_lanes:
                            cross_veh, cross_dist = cross_lane.prev_vehicle(0, route=nexti(cross_lane.from_routes))
                            human_remain = cross_veh and cross_veh.type is not rl_type
                        level = not human_remain and (horizontal_go and horizontal_lane or vertical_go and not horizontal_lane)
            elif c.act_type == 'accel':
                level = (np.clip(act, c.low, 1) - c.low) / (1 - c.low)
            else:
                level = act / (c.n_actions - 1)
            ts.accel(rl, (level * 2 - 1) * (c.max_accel if level > 0.5 else c.max_decel))

        if c.tl == 'MaxPressure':
            self.mp_tlast += c.sim_step
            tmin = c.get('mp_tmin', 0)
            if self.mp_tlast >= tmin:
                for tl in ts.traffic_lights:
                    if ts.get_program(tl) == 'off':
                        break
                    jun = tl.junction
                    pressures = [len(p.vehicles) - len(n.vehicles) for p, n in zip(jun.prev_lanes, jun.next_lanes)]

                    total_pressures = []
                    for phase in (ph for ph in tl.phases if 'y' not in ph.state):
                        total_pressures.append(sum(p for p, s in zip(pressures, phase.state) if s == 'G'))

                    ts.set_phase(tl, np.argmax(total_pressures))
                self.mp_tlast = 0

        super().step()
        self.build_platoon()

        obs = {}

        veh_default_close = Namespace(speed=max_speed, route_position=np.inf)
        veh_default_far = Namespace(speed=0, route_position=-np.inf)
        vehs_default = lambda: [veh_default_close] + [veh_default_far] * 2 * c.obs_next_cross_platoons
        for veh in rl_type.vehicles:
            route, lane, platoon = veh.route, veh.lane, veh.platoon
            junction = lane.next_junction

            head, tail = veh, platoon.tail
            route_vehs = [(route, [head, *lif(c.obs_tail, tail)])]

            if junction is ts.sentinel_junction:
                route_vehs.extend([(None, vehs_default())] * (len(c.directions) - 1))
            else:
                for jun_lane in lane.next_junction_lanes:
                    # Defaults for jun_lane
                    jun_headtails = vehs_default()

                    jun_lane_route = nexti(jun_lane.from_routes)
                    jun_veh, _ = jun_lane.prev_vehicle(0, route=jun_lane_route)
                    jun_veh = jun_veh if jun_veh and jun_veh.lane.next_junction is junction else None

                    if jun_veh:
                        if jun_veh.type is rl_type:
                            # If jun_veh is RL or jun_veh is human and there's no RL vehicle in front of it
                            jun_headtails[1: 3] = jun_veh, jun_veh.platoon.tail
                            platoon = jun_veh.platoon.prev
                            for i in 1 + 2 * np.arange(1, c.obs_next_cross_platoons):
                                if platoon is None: break
                                jun_headtails[i: i + 2] = platoon.head, platoon.tail
                                platoon = platoon.prev
                        else:
                            # If jun_veh is a human vehicle behind some RL vehicle (in another lane)
                            jun_headtails[0] = jun_veh.platoon.tail
                            next_cross_platoon = jun_veh.platoon.prev
                            if next_cross_platoon:
                                jun_headtails[1: 3] = next_cross_platoon.head, next_cross_platoon.tail
                                platoon = next_cross_platoon.prev
                                for i in 1 + 2 * np.arange(1, c.obs_next_cross_platoons):
                                    if platoon is None: break
                                    jun_headtails[i: i + 2] = platoon.head, platoon.tail
                                    platoon = platoon.prev
                    route_vehs.append((jun_lane_route, jun_headtails))

            dist_features, speed_features = [], []
            for route, vehs in route_vehs:
                j_pos = junction.route_position[route]
                dist_features.extend([0 if j_pos == v.route_position else (j_pos - v.route_position) / max_dist for v in vehs])
                speed_features.extend([v.speed / max_speed for v in vehs])

            obs[veh.id] = np.clip([*dist_features, *speed_features], 0, 1).astype(np.float32) * (1 - c.low) + c.low

        sort_id = lambda d: [v for k, v in sorted(d.items())]
        ids = sorted(obs)
        obs = arrayf(sort_id(obs)).reshape(-1, c._n_obs)
        if c.rew_type == 'outflow':
            reward = len(ts.new_arrived) - c.collision_coef * len(ts.new_collided)
        elif c.rew_type == 'time_penalty':
            reward = -c.sim_step * (len(ts.vehicles) + sum(len(f.backlog) for f in ts.flows)) - c.collision_coef * len(ts.new_collided)
        return Namespace(obs=obs, id=ids, reward=reward)

    def append_step_info(self):
        super().append_step_info()
        self.rollout_info.append(n_veh_network=len(self.ts.vehicles))

    @property
    def stats(self):
        c = self.c
        info = self.rollout_info[1 + c.warmup_steps + c.skip_stat_steps:]
        mean = lambda L: np.mean(L) if len(L) else np.nan
        stats = {**super().stats, **dif('length_range' in c, length=c.length), **dif('av_range' in c, av_frac=c.av_frac)}
        stats['backlog_step'] = mean(info['backlog'])
        stats['n_total_veh_step'] = mean(info['n_veh_network']) + stats['backlog_step']

        if c.multi_flowrate:
            stats['flow_horizontal'] = c.flow_rate_h
            stats['flow_vertical'] = c.flow_rate_v
        return stats

class Intersection(Main):
    def create_env(c):
        if c.multi_flowrate:
            return NormEnv(c, IntersectionEnv(c))
        else:
            c._norm = NormEnv(c, None)
            return IntersectionEnv(c)

    @property
    def observation_space(c):
        low = np.full(c._n_obs, c.low)
        return Box(low, np.ones_like(low))

    @property
    def action_space(c):
        if c.act_type == 'accel':
            return Box(low=c.low, high=1, shape=(1,), dtype=np.float32)
        else:
            return Discrete(c.n_actions)

    def on_rollout_end(c, rollout, stats, ii=None, n_ii=None):
        log = c.get_log_ii(ii, n_ii)
        step_obs_ = rollout.obs
        step_obs = step_obs_[:-1]

        if not c.multi_flowrate:
            rollout.raw_reward = rollout.reward
            rollout.reward = [c._norm.norm_reward(r) for r in rollout.raw_reward]

        ret, _ = calc_adv(rollout.reward, c.gamma)

        n_veh = np.array([len(o) for o in step_obs])
        step_ret = [[r] * nv for r, nv in zip(ret, n_veh)]
        rollout.update(obs=step_obs, ret=step_ret)

        step_id_ = rollout.pop('id')
        id = np.concatenate(step_id_[:-1])
        id_unique = np.unique(id)

        reward = np.array(rollout.pop('reward'))
        raw_reward = np.array(rollout.pop('raw_reward'))

        log(**stats)
        log(raw_reward_mean=raw_reward.mean(), raw_reward_sum=raw_reward.sum())
        log(reward_mean=reward.mean(), reward_sum=reward.sum())
        log(n_veh_step_mean=n_veh.mean(), n_veh_step_sum=n_veh.sum(), n_veh_unique=len(id_unique))
        return rollout

if __name__ == '__main__':
    c = Intersection.from_args(globals(), locals())
    c.setdefaults(
        n_steps=200,
        step_save=5,

        depart_speed=0,
        max_speed=13,
        max_dist=100,
        max_accel=1.5,
        max_decel=3.5,
        sim_step=0.5,
        generic_type=True,
        n_actions=3,

        adv_norm=False,
        batch_concat=True,

        render=False,

        warmup_steps=100,
        horizon=2000,
        directions='4way',
        av_frac=0.15,
        handcraft=False,
        handcraft_tl=None,
        flow_rate=700,
        length=100,
        n_rows=1,
        n_cols=1,
        speed_mode=SPEED_MODE.obey_safe_speed,

        act_type='accel_discrete',
        low=-1,

        alg=PG,
        n_gds=1,
        lr=1e-3,
        gamma=0.99,
        collision_coef=5, # If there's a collision, it always involves an even number of vehicles

        enter_length=False,
        short_exit=False,

        rew_type='outflow',
        norm_reward=True,
        center_reward=True,
        multi_flowrate=False,
        opt='RMSprop',

        obs_tail=True,
        obs_next_cross_platoons=1,
    )
    if c.directions == '4way':
        c.directions = ['up', 'right', 'down', 'left']
    c._n_obs = 2 * (1 + c.obs_tail + (1 + 2 * c.obs_next_cross_platoons) * (len(c.directions) - 1))
    assert c.get('use_critic', False) is False, 'Not supporting value functions yet'
    c.redef_sumo = 'length_range' in c
    c.run()