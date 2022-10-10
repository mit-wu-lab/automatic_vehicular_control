from lxml.etree import Element, SubElement, tostring, XMLParser
from xml.etree import ElementTree
import subprocess
import sumolib
import traci
import traci.constants as T # https://sumo.dlr.de/pydoc/traci.constants.html
from traci.exceptions import FatalTraCIError, TraCIException
import bisect
import warnings
import gym

from automatic_vehicular_control.u import *
from automatic_vehicular_control.ut import *

def val_to_str(x):
    return str(x).lower() if isinstance(x, bool) else str(x)

def dict_kv_to_str(x):
    return dict(map(val_to_str, kv) for kv in x.items())

def str_to_val(x):
    for f in int, float:
        try:
            return f(x)
        except (ValueError, TypeError):
            pass
    return x

def values_str_to_val(x):
    for k, v in x.items():
        x[k] = str_to_val(v)
    return x

class E(list):
    """
    Builder for lxml.etree.Element
    """
    xsi = Path('F') / 'resources' / 'xml' / 'XMLSchema-instance' # http://www.w3.org/2001/XMLSchema-instance
    root_args = dict(nsmap=dict(xsi=xsi))
    def __init__(self, _name, *args, **kwargs):
        assert all(isinstance(a, E) for a in args)
        super().__init__(args)
        self._dict = kwargs
        self._name = _name

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def __getitem__(self, k):
        if isinstance(k, (int, slice)):
            return super().__getitem__(k)
        return self._dict.__getitem__(k)

    def __setitem__(self, k, v):
        if isinstance(k, (int, slice)):
            return super().__setitem__(k, v)
        return self._dict.__setitem__(k, v)

    def __getattr__(self, k):
        if k == '__array_struct__':
            raise AttributeError
        if k in ['_dict', '_name']:
            return self.__dict__[k]
        else:
            return self[k]

    def __setattr__(self, k, v):
        if k in ['_dict', '_name']:
            self.__dict__[k] = v
        else:
            self[k] = v

    def __repr__(self):
        return self.to_string().decode()

    def to_element(self, root=True):
        e = Element(self._name, attrib={k: val_to_str(v) for k, v in self.items()}, **(E.root_args if root else {}))
        e.extend([x.to_element(root=False) for x in self])
        return e

    def to_string(self):
        return tostring(self.to_element(), pretty_print=True, encoding='UTF-8', xml_declaration=True)

    def to_path(self, p):
        p.save_bytes(self.to_string())

    def children(self, tag):
        return [x for x in self if x._name == tag]

    @classmethod
    def from_element(cls, e):
        return E(e.tag, *(cls.from_element(x) for x in e), **e.attrib)

    @classmethod
    def from_path(cls, p):
        return cls.from_element(ElementTree.parse(p, parser=XMLParser(recover=True)).getroot())

    @classmethod
    def from_string(cls, s):
        return cls.from_element(ElementTree.fromstring(s))

V = Namespace(**{k[4:].lower(): k for k, v in inspect.getmembers(T, lambda x: not callable(x)) if k.startswith('VAR_')})
TL = Namespace(**{k[3:].lower(): k for k, v in inspect.getmembers(T, lambda x: not callable(x)) if k.startswith('TL_')})

class SubscribeDef:
    """
    SUMO subscription manager
    """
    def __init__(self, tc_module, subs):
        self.tc_mod = tc_module
        self.names = [k.split('_', 1)[1].lower() for k in subs]
        self.constants = [getattr(T, k) for k in subs]

    def subscribe(self, *id):
        self.tc_mod.subscribe(*id, self.constants)
        return self

    def get(self, *id):
        res = self.tc_mod.getSubscriptionResults(*id)
        return Namespace(((n, res[v]) for n, v in zip(self.names, self.constants)))

SPEED_MODE = Namespace(
    aggressive=0,
    obey_safe_speed=1,
    no_collide=7,
    right_of_way=25,
    all_checks=31
)

LC_MODE = Namespace(off=0, no_lat_collide=512, strategic=1621)

# Traffic light defaults
PROGRAM_ID = 1
MAX_GAP = 3.0
DETECTOR_GAP = 0.6
SHOW_DETECTORS = True

# Car following models
IDM = dict(
    accel=2.6,
    decel=4.5,
    tau=1.0,  # past 1 at sim_step=0.1 you no longer see waves
    minGap=2.5,
    maxSpeed=30,
    speedFactor=1.0,
    speedDev=0.1,
    impatience=0.5,
    delta=4,
    carFollowModel='IDM',
    sigma=0.2,
)

Krauss = dict(
    accel=2.6,
    decel=4.5,
    tau=1.0,
    minGap=2.5,
    sigma=0.5,
    maxSpeed=30,
    speedFactor=1.0,
    speedDev=0.1,
    impatience=0.5,
    carFollowModel='Krauss',
)

# Lane change models
LC2013 = dict(
    laneChangeModel='LC2013',
    lcStrategic=1.0,
    lcCooperative=1.0,
    lcSpeedGain=1.0,
    lcKeepRight=1.0,
)

SL2015 = dict(
    laneChangeModel='SL2015',
    lcStrategic=1.0,
    lcCooperative=1.0,
    lcSpeedGain=1.0,
    lcKeepRight=1.0,
    lcLookAheadLeft=2.0,
    lcSpeedGainRight=1.0,
    lcSublane=1.0,
    lcPushy=0,
    lcPushyGap=0.6,
    lcAssertive=1,
    lcImpatience=0,
    lcTimeToImpatience=float('inf'),
    lcAccelLat=1.0,
)

# Builder for an inflow
def FLOW(id, type, route, departSpeed, departLane='random', vehsPerHour=None, probability=None, period=None, number=None):
    flow = Namespace(
        id=id,
        type=type,
        route=route,
        departSpeed=departSpeed,
        departLane=departLane,
        begin=1
    )
    flow.update(dict(number=number) if number else dict(end=86400))
    if vehsPerHour:
        flow.vehsPerHour = vehsPerHour
    elif probability:
        flow.probability = probability
    elif period:
        flow.period = period
    return flow

# Vehicle colors
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 0, 0)

COLLISION = Namespace(teleport='teleport', warn='warn', none='none', remove='remove')

class NetBuilder:
    """
    Builder for the traffic network, which includes nodes, edges, connections, and additional
    Output can be saved into xml and serve as input to SUMO
    """
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.connections = {}
        self.additional = []

    def add_nodes(self, node_infos):
        nodes = [E('node', **n.setdefaults(id=f'n_{n.x}.{n.y}')) for n in node_infos]
        self.nodes.update((n.id, n) for n in nodes)
        ret = np.empty(len(nodes), dtype=object)
        ret[:] = nodes
        return ret

    def chain(self, nodes, lane_maps=None, edge_attrs={}, route_id=None):
        """
        Add a chain of nodes while building the edges, connections, and routes
        """
        edge_attrs = [edge_attrs] * (len(nodes) - 1) if isinstance(edge_attrs, dict) else edge_attrs
        lane_maps = lane_maps or [{0: 0} for _ in range(len(nodes) - 2)]
        num_lanes = ([len(l) for l in lane_maps] + [len(set(lane_maps[-1].values()))])
        edges = [E('edge', **{
            'id': f'e_{n1.id}_{n2.id}',
            'from': n1.id, 'to': n2.id,
            'length': np.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2),
            'numLanes': nl,
            **e
        }) for n1, n2, nl, e in zip(nodes, nodes[1:], num_lanes, edge_attrs)]
        connections = flatten([[E('connection', **{
            'from': e1.id, 'to': e2.id,
            'fromLane': from_, 'toLane': to
        }) for from_, to in lmap.items()] for e1, e2, lmap in zip(edges, edges[1:], lane_maps)])
        route = E('route', id=route_id or f'r_{len(self.additional)}', edges=' '.join(e.id for e in edges))
        self.edges.update((e.id, e) for e in edges)
        self.connections.update(((con['from'], con.to, con.fromLane, con.toLane), con) for con in connections)
        self.additional.append(route)
        return edges, connections, route

    def intersect(self, center, neighbors, movements, separate_right=False):
        """
        Assume no u-turns. Number of output lanes is the max number of lanes of all the movements leading to the output. Right turns either share lane with straight or have one separate lane.
        neighbors: n neighboring nodes of center, specified in a clockwise orientation from the top
        movements: array of shape (4, 4) where entry (i, j) denotes number of lanes from ith node to jth node. i = 0 denotes the top node, and i increases clockwise around the intersection
        separate_right: a boolean or an array of shape (n,) which denotes whether the right turn should be a separate lane or share the lane going straight
        """
        turn_nodes, turn_lanes = movements_to_turns(neighbors, movements)
        opp_turn_lanes = np.roll(turn_lanes, 2, axis=0)
        separate_right = [separate_right] * len(neighbors) if isinstance(separate_right, bool) else separate_right
        edges = []
        for x, num_x_to, num_opp_to, num_to_x, sep in zip(neighbors, turn_lanes, opp_turn_lanes, movements.T, separate_right):
            pairs = []
            p1, p2 = np.array([x.x, x.y]), np.array([center.x, center.y])
            d = p2 - p1
            dist = np.linalg.norm(d)
            d1, d2 = np.array([-d[1], d[0]]) / dist * 3.2 * min(num_x_to[-1], num_opp_to[-1]) * 0.5 # 3.2 is the lane width
            if num_x_to.any(): pairs.append((x, center, num_x_to.sum() - min(num_x_to[0], not sep)))
            if num_to_x.any(): pairs.append((center, x, num_to_x.max()))
            for x1, x2, n_lanes in pairs:
                edges.append(E('edge', **{
                    'id': f'e_{x1.id}_{x2.id}',
                    'from': x1.id, 'to': x2.id,
                    'length': dist, 'shape': f'{x1.x + d1},{x1.y + d2} {x2.x + d1},{x2.y + d2}',
                    'numLanes': n_lanes,
                }))
        self.edges.update((e.id, e) for e in edges)

        connections = []
        for x1, x1_to, num_x1_to, sep in zip(neighbors, turn_nodes, turn_lanes, separate_right):
            offsets1 = np.cumsum(num_x1_to) - num_x1_to
            offsets1[1:] = offsets1[1:] - min(num_x1_to[0], not sep)
            e1 = f'e_{x1.id}_{center.id}'
            for i, (x2, num_x1_to_x2, o1) in enumerate(zip(x1_to, num_x1_to, offsets1)):
                e2 = f'e_{center.id}_{x2.id}'
                o2 = 0 if i == 0 else self.edges[e2].numLanes - num_x1_to_x2 # 0 for right turns
                connections.extend(E('connection', **{'from': e1, 'to': e2, 'fromLane': o1 + k, 'toLane': o2 + k}) for k in range(num_x1_to_x2))
        self.connections.update(((con['from'], con.to, con.fromLane, con.toLane), con) for con in connections)

        routes = []
        for x1, x1_to in zip(neighbors, turn_nodes):
            e1, r1 = f'e_{x1.id}_{center.id}', f'r_{x1.id}_{center.id}'
            routes.append(E('route', id=r1, edges=e1))
            routes1 = [E('route', id=f'r_{x1.id}_{x2.id}', edges=f'{e1} e_{center.id}_{x2.id}') for x2 in x1_to]
            routes.extend(routes1)

            routes.append(E('rerouter', E('interval',
                *(E('routeProbReroute', id=r.id, probability=1) for r in routes1), begin=0, end=1e9)
            , id='reroute', edges=e1))
        self.additional.extend(routes)
        return edges, connections, routes

    def build(self):
        return E('nodes', *self.nodes.values()), E('edges', *self.edges.values()), E('connections', *self.connections.values()), self.additional

def movements_to_turns(nodes, movements):
    """
    nodes: array of shape (4,) nodes
    movements: array of shape (4, 4) where entry (i, j) denotes number of lanes from ith node to jth node. i = 0 denotes the top node, and i increases clockwise around the intersection
    return:
        - array of shape (4, 3) where row i denotes the right, straight, and left nodes of node i
        - array of shape (4, 3) where row i denotes number of lanes turning right, going straight, and turning left, respectively.
    """
    return (
        [(nodes[i + 1:] + nodes[:i])[::-1] for i, _ in enumerate(nodes)],
        np.array([np.roll(ms, -i)[:0:-1] for i, ms in enumerate(movements)])
    )

def build_closed_route(edges, n_veh=0, av=0, space='random_free', type_fn=None, depart_speed=0, offset=0, init_length=None):
    """
    'equal' and 'free' guarantee the order of the vehicle ids to be monotonically increasing along the route.
    Anything with random could see random vehicle id order along the route
    """
    assert isinstance(space, (float, int)) or space in ('equal', 'random_free', 'free', 'random', 'base', 'last')
    order = lambda i: edges[i:] + edges[:i + 1]
    routes = [E('route', id=f'route_{e.id}', edges=' '.join(e_.id for e_ in order(i))) for i, e in enumerate(edges)]
    rerouter = E('rerouter', E('interval', E('routeProbReroute', id=routes[0].id), begin=0, end=1e9), id='reroute', edges=edges[0].id)
    vehicles = []
    if n_veh > 0:
        lane_lengths, lane_routes, lane_idxs = map(np.array, zip(*[(e.length, r.id, i) for e, r in zip(edges, routes) for i in range(e.numLanes)]))
        lane_ends = np.cumsum(lane_lengths)
        lane_starts = lane_ends - lane_lengths
        total_length = lane_ends[-1]
        init_length = init_length or total_length

        positions = (offset + np.linspace(0, init_length, n_veh, endpoint=False)) % total_length
        veh_lane = (positions.reshape(-1, 1) < lane_ends.reshape(1, -1)).argmax(axis=1)

        if space == 'equal':
            space = total_length / n_veh
        if isinstance(space, (float, int)):
            veh_lane_pos = positions - lane_starts[veh_lane]
        else:
            veh_lane_pos = [space] * n_veh
        veh_routes = lane_routes[veh_lane]
        veh_lane_idxs = lane_idxs[veh_lane]
        type_fn = type_fn or (lambda i: 'rl' if i < av else 'human')
        vehicles = [E('vehicle', id=f'{i}', type=type_fn(i), route=r, depart='0', departPos=p, departLane=l, departSpeed=depart_speed) for i, (r, p, l) in enumerate(zip(veh_routes, veh_lane_pos, veh_lane_idxs))]
    return [*routes, rerouter, *vehicles]

class SumoDef:
    """
    Given the network definitions in Python
    1. Save them into xml files for SUMO
    2. Run SUMO's netconvert command on the input files
    3. Start the SUMO simulation as a subprocess
    """
    no_ns_attr = '{%s}noNamespaceSchemaLocation' % E.xsi
    xsd = Path.env('F') / 'resources' / 'xml' / '%s_file.xsd' # http://sumo.dlr.de/xsd/%s_file.xsd For all other defs except viewsettings
    config_xsd = Path.env('F') / 'resources' / 'xml' / 'sumoConfiguration.xsd' # http://sumo.dlr.de/xsd/sumoConfiguration.xsd
    # see https://sumo.dlr.de/docs/NETCONVERT.html
    netconvert_args = dict(nodes='n', edges='e', connections='x', types='t')
    config_args = dict(
        net='net-file', routes='route-files',
        additional='additional-files', gui='gui-settings-file'
    )
    file_args = set(netconvert_args.keys()) | set(config_args.keys())

    def __init__(self, c):
        self.c = c
        self.dir = c.res.rel() / 'sumo'
        if 'i_worker' in c:
            self.dir /= c.i_worker
        self.dir.mk() # Use relative path here to shorten sumo arguments
        self.sumo_cmd = None

    def save(self, *args, **kwargs):
        for e in args:
            e[SumoDef.no_ns_attr] = SumoDef.xsd % e._name
            kwargs[e._name] = path = self.dir / e._name[:3] + '.xml'
            e.to_path(path)
        return Namespace(**kwargs)

    def generate_net(self, **kwargs):
        net_args = Namespace(**kwargs.get('net_args', {})).setdefaults(**{
            'no-turnarounds': True
        })

        # https://sumo.dlr.de/docs/NETCONVERT.html
        dyld_env_var = os.environ.get('DYLD_LIBRARY_PATH')
        net_path = self.dir / 'net.xml'
        args = [*lif(dyld_env_var, f'DYLD_LIBRARY_PATH={dyld_env_var}'), 'netconvert', '-o', net_path]
        for name, arg in SumoDef.netconvert_args.items():
            path = kwargs.pop(name, None)
            if path:
                args.append('-%s %s' % (arg, path))
        args.extend('--%s %s' % (k, val_to_str(v)) for k, v in net_args.items())

        cmd = ' '.join(args)
        self.c.log(cmd)
        out, err = shell(cmd, stdout=None)
        if err:
            self.c.log(err)

        return net_path

    def generate_sumo(self, **kwargs):
        c = self.c

        gui_path = self.dir / 'gui.xml'
        E('viewsettings',
            E('scheme', name='real world'),
            E('background',
                backgroundColor='150,150,150',
                showGrid='0',
                gridXSize='100.00',
                gridYSize='100.00'
            )).to_path(gui_path)
        kwargs['gui'] = gui_path

        # https://sumo.dlr.de/docs/SUMO.html
        sumo_args = Namespace(
            **{arg: kwargs[k] for k, arg in SumoDef.config_args.items() if k in kwargs},
            **kwargs.get('sumo_args', {})).setdefaults(**{
            'begin': 0,
            # 'num-clients': 1,
            'step-length': c.sim_step,
            'no-step-log': True,
            'time-to-teleport': -1,
            'no-warnings': c.get('no_warnings', True),
            'collision.action': COLLISION.remove,
            'collision.check-junctions': True,
            'max-depart-delay': c.get('max_depart_delay', 0.5),
            'random': True,
            'start': c.get('start', True)
        })
        cmd = ['sumo-gui' if c.render else 'sumo']
        for k, v in sumo_args.items():
            cmd.extend(['--%s' % k, val_to_str(v)] if v is not None else [])
        c.log(' '.join(cmd))
        return cmd

    def start_sumo(self, tc, tries=3):
        for _ in range(tries):
            try:
                if tc and not 'TRACI_NO_LOAD' in os.environ:
                    tc.load(self.sumo_cmd[1:])
                else:
                    if tc:
                        tc.close()
                    else:
                        self.port = sumolib.miscutils.getFreeSocketPort()
                    # Taken from traci.start but add the DEVNULL here
                    p = subprocess.Popen(self.sumo_cmd + ['--remote-port', f'{self.port}'], **dif(self.c.get('sumo_no_errors', True), stderr=subprocess.DEVNULL))
                    tc = traci.connect(self.port, 10, 'localhost', p)
                return tc
            except traci.exceptions.FatalTraCIError: # Sometimes there's an unknown error while starting SUMO
                if tc:
                    tc.close()
                self.c.log('Restarting SUMO...')
                tc = None

class Namespace(Namespace):
    """
    A wrapper around dictionary with nicer formatting for the Entity subclasses
    """
    @classmethod
    def format(cls, x):
        def helper(x):
            if isinstance(x, Entity):
                return f"{type(x).__name__}('{x.id}')"
            elif isinstance(x, (list, tuple, set, np.ndarray)):
                return [helper(y) for y in x]
            elif isinstance(x, np.generic):
                return x.item()
            elif isinstance(x, dict):
                return {helper(k): helper(v) for k, v in x.items()}
            return x
        return format_yaml(helper(x))

    def __repr__(self):
        return Namespace.format(self)

class Container(Namespace):
    def __iter__(self):
        return iter(self.values())

class Entity(Namespace):
    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return self.id

    def __repr__(self):
        inner_content = Namespace.format(dict(self)).replace('\n', '\n  ').rstrip(' ')
        return f"{type(self).__name__}('{self.id}',\n  {inner_content})\n\n"

class Vehicle(Entity):
    def leader(self, use_edge=False, use_route=True, max_dist=np.inf):
        try:
            return next(self.leaders(use_edge, use_route, max_dist, 1))
        except StopIteration:
            return None, 0

    def leaders(self, use_edge=False, use_route=True, max_dist=np.inf, max_count=np.inf):
        ent, i = (self.edge, self.edge_i) if use_edge else (self.lane, self.lane_i)
        route = self.route if use_route else None
        for veh, dist in ent.next_vehicles_helper(i + 1, route, max_dist + self.laneposition, max_count):
            yield veh, dist - self.laneposition

    def follower(self, use_edge=False, use_route=True, max_dist=np.inf):
        try:
            return next(self.followers(use_edge, use_route, max_dist, 1))
        except StopIteration:
            return None, 0

    def followers(self, use_edge=False, use_route=True, max_dist=np.inf, max_count=np.inf):
        ent, i = (self.edge, self.edge_i) if use_edge else (self.lane, self.lane_i)
        route = self.route if use_route else None
        for veh, dist in ent.prev_vehicles_helper(i - 1, route, max_dist - self.laneposition, max_count):
            yield veh, dist + self.laneposition

class Type(Entity): pass
class Flow(Entity): pass
class Junction(Entity): pass

class Edge(Entity):
    def next(self, route):
        return route.next.get(self)
    def prev(self, route):
        return route.prev.get(self)

    def find(self, position, route):
        """
        Find the entity where the position belongs in and return the relative position to that entity
        position: relative position to the base of self
        return:
            position: the relative position to ent within [0, ent.length)
            offset: the offset from self to ent
        """
        ent = self
        offset = 0
        while position < 0 and ent and route:
            ent = ent.prev(route)
            position += ent.length
            offset -= ent.length
        while position >= ent.length and ent and route:
            position -= ent.length
            offset += ent.length
            ent = ent.next(route)
        assert ent is None or 0 <= position < ent.length
        return ent, position, offset

    def next_vehicle(self, position, route=None, max_dist=np.inf, filter=lambda veh: True):
        try:
            return next(self.next_vehicles(position, route, max_dist, 1, filter))
        except StopIteration:
            return None, 0

    def prev_vehicle(self, position, route=None, max_dist=np.inf, filter=lambda veh: True):
        try:
            return next(self.prev_vehicles(position, route, max_dist, 1, filter))
        except StopIteration:
            return None, 0

    def next_vehicles(self, position, route=None, max_dist=np.inf, max_count=np.inf, filter=lambda veh: True):
        """
        yield:
            veh: the closest veh at a higher position
            offset: veh.position - position
        """
        assert max_dist == np.inf or position == 0, 'Haven\'t implemented general case yet'
        ent, position, offset = self.find(position, route) if route else (self, position, 0)
        i = bisect.bisect_left(ent.positions, position)
        for veh, dist in ent.next_vehicles_helper(i, route, max_dist, max_count, filter):
            yield veh, dist - position

    def prev_vehicles(self, position, route=None, max_dist=np.inf, max_count=np.inf, filter=lambda veh: True):
        """
        yield:
            veh: the closest veh at a lower position
            offset: position - veh.position
        """
        assert max_dist == np.inf or position == 0, 'Haven\'t implemented general case yet'
        ent, position, offset = self.find(position, route) if route else (self, position, 0)
        i = bisect.bisect_right(ent.positions, position)
        for veh, dist in ent.prev_vehicles_helper(i - 1, route, max_dist, max_count, filter):
            yield veh, dist + position

    def next_vehicles_helper(self, i, route=None, max_dist=np.inf, max_count=np.inf, filter=lambda veh: True):
        """ Iterate over next vehicles starting from the ith vehicle and yield veh and distance from base of edge / lane. Keep searching along the route if route is specified, otherwise only search this edge or lane
        yield:
            dist: vehicle position - base position
        """
        while i < len(self.vehicles):
            veh = self.vehicles[i]
            veh_dist = veh.laneposition
            if veh_dist > max_dist: return
            if filter(veh): yield veh, veh_dist
            max_count -= 1
            if max_count <= 0: return
            i += 1
        dist = self.length
        if route and dist <= max_dist:
            ent = self.next(route)
            while ent is not None:
                for veh in ent.vehicles:
                    veh_dist = dist + veh.laneposition
                    if veh_dist > max_dist: return
                    if filter(veh): yield veh, veh_dist
                    else: continue
                    max_count -= 1
                    if max_count <= 0: return
                ent = ent.next(route)

    def prev_vehicles_helper(self, i, route=None, max_dist=np.inf, max_count=np.inf, filter=lambda veh: True):
        """ Iterate prev vehicle starting from the ith vehicle and yield veh and offset (could be negative) from base of edge / lane. Keep searching along the route if route is specified, otherwise only search this edge or lane
        yield:
            dist: base position - vehicle position
        """
        while i >= 0:
            veh = self.vehicles[i]
            veh_dist = -veh.laneposition
            if veh_dist > max_dist: return
            if filter(veh): yield veh, veh_dist
            max_count -= 1
            if max_count <= 0: return
            i -= 1
        dist = 0
        if route and dist <= max_dist:
            ent = self.prev(route)
            while ent is not None:
                dist += ent.length
                for veh in reversed(ent.vehicles):
                    veh_dist = dist - veh.laneposition
                    if veh_dist > max_dist: return
                    if filter(veh): yield veh, veh_dist
                    else: continue
                    max_count -= 1
                    if max_count <= 0: return
                ent = ent.prev(route)

class Lane(Edge): pass

class Route(Entity):
    def next_edges(self, edge):
        return self.edges[self.edges.index(edge) + 1:]

    def prev_edges(self, edge):
        return self.edges[self.edges.index(edge) - 1::-1]

    def prev_vehicle(self, route_position):
        i = bisect.bisect(self.positions, route_position)
        return self.vehicles[i - 1] if i > 0 else None

    def next_vehicle(self, route_position):
        i = bisect.bisect(self.positions, route_position)
        return self.vehicles[i + 1] if i + 1 < len(self.vehicles) else None

class TrafficState:
    """
    Keeps relevant parts of SUMO simulation state in Python for easy access
    """
    def __init__(self, c, tc, net, **kwargs):
        """
        Initialize and populate container objects
        """
        self.c = c
        self.tc = tc
        # https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html
        # https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html
        self.edges = edges = Container()
        self.lanes = lanes = Container()
        for e_edge in map(values_str_to_val, net.children('edge')):
            edges[e_edge.id] = edge = Edge(**e_edge,
                lanes=[None] * len(e_edge), route_position=Namespace(),
                vehicles=[], positions=[],
                froms=set(), tos=set(),
                from_routes=set(), to_routes=set())
            for e_lane in map(values_str_to_val, e_edge):
                lanes[e_lane.id] = lane = Lane(**e_lane,
                    edge=edge,
                    froms=set(), tos=set(),
                    vehicles=[], positions=[],
                    from_routes=set(), to_routes=set())
                edge.lanes[int(e_lane['index'])] = lane
            for i, lane in enumerate(edge.lanes):
                lane.left = edge.lanes[i - 1] if i > 0 else None
                lane.right = edge.lanes[i + 1] if i < len(edge.lanes) - 1 else None
            edge.length = lane.length
        self.internal_edges = Container((k, v) for k, v in edges.items() if k.startswith(':'))
        self.internal_lanes = Container((k, v) for k, v in lanes.items() if k.startswith(':'))
        self.external_edges = Container((k, v) for k, v in edges.items() if not k.startswith(':'))
        self.external_lanes = Container((k, v) for k, v in lanes.items() if not k.startswith(':'))

        segments = {}
        def add_from_to(from_, to):
            from_.tos.add(to)
            to.froms.add(from_)
        for con in map(values_str_to_val, net.children('connection')):
            from_lane = edges[con['from']].lanes[int(con.fromLane)]
            to_lane = edges[con.to].lanes[int(con.toLane)]
            if 'via' in con.keys(): # connect via internal lane
                via_lane = lanes[con.via]
                add_from_to(from_lane, via_lane)
                add_from_to(via_lane, to_lane)
                add_from_to(from_lane.edge, via_lane.edge)
                add_from_to(via_lane.edge, to_lane.edge)
                segments[from_lane.edge, to_lane.edge] = (from_lane.edge, via_lane.edge, to_lane.edge)
            else:
                add_from_to(from_lane, to_lane)
                add_from_to(from_lane.edge, to_lane.edge)
                segments[from_lane.edge, to_lane.edge] = (from_lane.edge, to_lane.edge)

        self.junctions = junctions = Container()
        for e_jun in map(values_str_to_val, net.children('junction')):
            if e_jun.type in ['dead_end']:
                continue
            junctions[e_jun.id] = jun = Junction(**e_jun)
            jun.prev_lanes = [lanes[lane_id] for lane_id in jun.incLanes.split(' ')]
            jun.lanes = [lanes[lane_id] for lane_id in jun.intLanes.split(' ')]
            jun.next_lanes = [next_lane for lane in jun.lanes for next_lane in lane.tos]
            for i, (lane, e_resp) in enumerate(zip(jun.lanes, e_jun)):
                lane.junction = jun
                lane.junction_lanes = jun.lanes[i + 1:] + jun.lanes[:i] # The other lanes at the junction, in a clockwise direction
                lane.cross_lanes = set(lane for lane, foe in zip(jun.lanes, reversed(e_resp.foes)) if foe == '1') # use foes instead of response here because intersections have priority, but we don't want priority
            jun.route_position = Namespace() # Maps a route to the position of this junction along that route
        self.sentinel_junction = Junction(id=None, route_position=defaultdict(lambda: np.inf))

        # https://sumo.dlr.de/docs/Simulation/Rerouter.html
        add_defs = [kwargs[k] for k in ['routes', 'additional'] if k in kwargs]
        add_children = lambda tag: itertools.chain(*(x.children(tag) for x in add_defs))
        self.types = types = Container()
        for v_type in map(values_str_to_val, add_children('vType')):
            types[v_type.id] = Type(**v_type, vehicles=set())

        self.routes = Container()
        for e_route in map(values_str_to_val, add_children('route')):
            self.routes[e_route.id] = route = Route(**e_route)
            route_external_edges = [edges[id] for id in e_route.edges.split(' ')]
            route.next = Namespace() # Maps edge in the route to the next edge in the route
            route.prev = Namespace()
            route.edges = [route_external_edges[0]]
            for e1, e2 in zip(route_external_edges, route_external_edges[1:]):
                segment = segments[e1, e2]
                e1.to_routes.add(route)
                e2.from_routes.add(route)
                for curr, next_ in zip(segment, segment[1:]):
                    route.next[curr] = next_
                    route.prev[next_] = curr
                    for curr_lane in curr.lanes:
                        curr_lane.to_routes.add(route)
                        for to_lane in curr_lane.tos:
                            if to_lane.edge is next_:
                                assert curr_lane in to_lane.froms
                                route.next[curr_lane] = to_lane
                                route.prev[to_lane] = curr_lane
                                to_lane.from_routes.add(route)
                                break
                route.edges.extend(segment[1:])

            route.junctions = junctions, junction_positions = [], []
            offset = 0
            for edge in route.edges:
                edge.route_position.setdefault(route, offset) # If already set, don't set again if same edge appears multiple times

                lane = nexti(edge.lanes)
                junction = lane.get('junction')
                if junction:
                    junctions.append(junction)
                    junction_positions.append(offset)
                    junction.route_position[route] = offset
                offset += edge.length

            next_junction = self.sentinel_junction
            next_junction_lanes, next_cross_lanes = set(), set()
            for edge in reversed(route.edges):
                lane = nexti(edge.lanes)
                lane.next_junction = next_junction
                lane.next_junction_lanes = next_junction_lanes
                lane.next_cross_lanes = next_cross_lanes
                junction = lane.get('junction')
                if junction:
                    next_junction = junction
                    next_cross_lanes = lane.cross_lanes
                    next_junction_lanes = lane.junction_lanes

        self.flows = flows = Container()
        for e_flow in map(values_str_to_val, add_children('flow')):
            flows[e_flow.id] = flow = Flow(**e_flow)
            flow.route = route = self.routes[e_flow.route]
            flow.edge = route.edges[0]
            flow.type = self.types.get(e_flow.type)
            flow.backlog = set() # IDs of backlogged vehicles for this inflow
            # For generic_type only
            flow.count = 0
            flow.last_rl = False

        self.subscribes = Namespace()

        # https://sumo.dlr.de/docs/TraCI/Object_Variable_Subscription.html
        self.traffic_lights = traffic_lights = Container()
        for tl in map(values_str_to_val, add_children('tlLogic')):
            traffic_lights[tl.id] = Entity(**tl,
                junction=self.junctions[tl.id],
                phases=[Namespace(x) for x in tl]
            )

        self.vehicles = Container()
        self.new_arrived = self.new_departed = self.new_collided = None # Sets of vehicles
        self.all_arrived, self.all_departed, self.all_collided = [], [], []

    def setup(self):
        """
        Add subscriptions for some SUMO state variables
        """
        tc = self.tc
        subscribes = self.subscribes

        subscribes.sim = SubscribeDef(tc.simulation, [
            V.departed_vehicles_ids, V.arrived_vehicles_ids,
            V.colliding_vehicles_ids, V.loaded_vehicles_ids]).subscribe()
        subscribes.tl = SubscribeDef(tc.trafficlight, [
            TL.red_yellow_green_state])
        subscribes.veh = SubscribeDef(tc.vehicle, [
            V.road_id, V.lane_index, V.laneposition,
            V.speed, V.position, V.angle,
            V.fuelconsumption, V.noxemission])
        for tl_id in self.traffic_lights.keys():
            subscribes.tl.subscribe(tl_id)

    def compute_type(self, veh_id):
        """
        If using 'generic' as the vehicle type, dynamically assign the vehicle type after the vehicle inflows
        """
        c = self.c
        av_frac = c.av_frac
        if '.' not in veh_id: # initialized vehicle but not from flow
            return 'human'
        flow_id, _ = veh_id.rsplit('.')
        flow = self.flows[flow_id]
        if type(c.generic_type) in (float, int): # Cannot use isinstance here since we don't want True to land in this branch
            assert av_frac == 0.5
            keep_prob = c.generic_type
            flow.last_rl = is_rl = np.random.rand() < (keep_prob if flow.last_rl else (1 - keep_prob))
        elif isinstance(c.generic_type, tuple):
            assert av_frac == 0.5
            keep_prob_rl, keep_prob_human = c.generic_type
            flow.last_rl = is_rl = np.random.rand() < (keep_prob_rl if flow.last_rl else (1 - keep_prob_human))
        elif c.generic_type == 'rand':
            is_rl = np.random.rand() < av_frac
        else:
            flow.count = veh_index = flow.count + 1
            cycle = int(np.ceil(veh_index * av_frac))
            last_cycle = int(np.ceil((veh_index - 1) * av_frac))
            is_rl = cycle > last_cycle
        return 'rl' if is_rl else 'human'

    def step(self):
        """
        Take a simulation step and update state
        """
        c = self.c
        tc = self.tc
        subscribes = self.subscribes

        # Actual SUMO step
        tc.simulationStep()
        sim_res = subscribes.sim.get()

        for veh in self.vehicles:
            veh.unvar('edge', 'lane')
        for ent in itertools.chain(self.edges, self.lanes):
            ent.vehicles.clear()
            ent.positions.clear()

        for tl_id, tl in self.traffic_lights.items():
            tl.update(subscribes.tl.get(tl_id))

        for veh_id in sim_res.loaded_vehicles_ids:
            flow_id, _ = veh_id.rsplit('.')
            flow = self.flows[flow_id]
            flow.backlog.add(veh_id)

        self.new_departed = set()
        for veh_id in sim_res.departed_vehicles_ids:
            subscribes.veh.subscribe(veh_id)
            type_id = tc.vehicle.getTypeID(veh_id)
            if type_id == 'generic':
                type_id = self.compute_type(veh_id)
            type_ = self.types[type_id]
            route = self.routes[tc.vehicle.getRouteID(veh_id)]
            length = tc.vehicle.getLength(veh_id)
            self.vehicles[veh_id] = veh = Vehicle(id=veh_id, type=type_, route=route, length=length)
            type_.vehicles.add(veh)

            if c.render:
                color_fn = c.get('color_fn', lambda veh: RED if 'rl' in type_.id else WHITE)
                self.set_color(veh, color_fn(veh))

            tc.vehicle.setSpeedMode(veh_id, c.get('speed_mode', SPEED_MODE.all_checks))
            tc.vehicle.setLaneChangeMode(veh_id, c.get('lc_mode', LC_MODE.no_lat_collide))
            self.new_departed.add(veh)
            if '.' in veh_id:
                flow_id, _ = veh_id.rsplit('.')
                if flow_id in self.flows:
                    flow = self.flows[flow_id]
                    flow.backlog.remove(veh_id)

        self.new_arrived = {self.vehicles[veh_id] for veh_id in sim_res.arrived_vehicles_ids}
        self.new_collided = {self.vehicles[veh_id] for veh_id in sim_res.colliding_vehicles_ids}
        for veh in self.new_arrived:
            veh.type.vehicles.remove(self.vehicles.pop(veh.id))
        self.new_arrived -= self.new_collided # Don't count collided vehicles as "arrived"

        for veh_id, veh in self.vehicles.items():
            veh.prev_speed = veh.get('speed', None)
            veh.update(subscribes.veh.get(veh_id))
            edge = self.edges[veh.road_id]
            edge.vehicles.append(veh)
            veh.edge = edge

        for edge in self.edges:
            edge.vehicles.sort(key=lambda veh: veh.laneposition)
            edge.positions = [veh.laneposition for veh in edge.vehicles]
            for edge_i, veh in enumerate(edge.vehicles):
                veh.edge_i = edge_i
                veh.lane = lane = edge.lanes[veh.lane_index]
                veh.lane_i = len(lane.vehicles)
                lane.vehicles.append(veh)
                lane.positions.append(veh.laneposition)

        self.all_arrived.append(self.new_arrived)
        self.all_departed.append(self.new_departed)
        self.all_collided.append(self.new_collided)

    def reset(self, tc):
        self.tc = tc
        self.subscribes.clear()
        self.vehicles.clear()
        self.all_departed, self.all_arrived = [], []
        self.new_arrived = self.new_departed = self.new_collided = None
        for type_ in self.types:
            type_.vehicles.clear()
        for flow in self.flows:
            flow.count = 0
            flow.backlog.clear()
        for ent in itertools.chain(self.edges, self.lanes):
            ent.vehicles.clear()
            ent.positions.clear()

    """ Wrapper methods for traci calls to interact with simulation """

    def remove(self, veh_id):
        try:
            self.tc.vehicle.remove(veh_id)
            self.tc.vehicle.unsubscribe(veh_id)
        except TraCIException as e:
            warnings.warn('Received nonfatal error while removing vehicle %s:\n%s' % (veh_id, e))

    def add(self, veh_id, route, type, lane_index='first', pos='base', speed=0, patience=3):
        try:
            self.tc.vehicle.add(veh_id, str(route.id), typeID=str(type.id),
                departLane=str(lane_index), departPos=str(pos), departSpeed=str(speed))
        except TraCIException as e:
            if patience == 0:
                raise FatalTraCIError('Tried for 3 times to add vehicle but still got error: ' + str(e))
            warnings.warn('Caught the following exception while adding vehicle %s, removing and readding vehicle:\n%s' % (veh_id, e))
            self.remove(veh_id)
            self.add(veh_id, route, type, lane_index, pos, speed, patience=patience - 1)

    def set_color(self, veh, color):
        self.tc.vehicle.setColor(veh.id, color + (255,))

    def accel(self, veh, acc, n_acc_steps=1):
        """
        Let the initial speed be v0, the sim_step be dt, and the acceleration be a. This function increases v0 over n=n_acc_steps steps by a*dt/n per step. At each of the sim steps, the speed increases by a*dt/n at the BEGINNING of the step. After one step, the vehicle's speed is v1=v0+a*dt/n and the distance traveled is v1*dt. If n>1, then after two steps, the vehicle's speed is v2=v1+a*dt/n and the distance traveled is v2*dt. Etc etc. If accel is called again before n steps has elapsed, the new acceleration action overrides the continuation of any previous acceleration. The per step acceleration a/n is clipped by SUMO's IDM behavior to be in the range of -max_decel <= a/n <= max_accel, where max_accel and max_decel are the IDM parameters given to SUMO.
        """
        veh.accel = acc
        self.tc.vehicle.slowDown(veh.id, max(0, veh.speed + acc * self.c.sim_step), (n_acc_steps - 1) * self.c.sim_step)

    def set_max_speed(self, veh, speed):
        self.tc.vehicle.setMaxSpeed(veh.id, max(speed, 1e-3))

    def lane_change(self, veh, direction):
        assert direction in [-1, 0, 1]
        veh.lane_change = lane_index = veh.lane_index + int(direction)
        self.tc.vehicle.changeLane(veh.id, lane_index, 100000.0)

    def lane_change_to(self, veh, lane_index):
        veh.lane_change = lane_index
        self.tc.vehicle.changeLane(veh.id, lane_index, 100000.0)

    def set_program(self, tl, program):
        self.tc.trafficlight.setProgram(tl.id, program)

    def get_program(self, tl):
        return self.tc.trafficlight.getProgram(tl.id)

    def set_phase(self, tl, phase_index):
        return self.tc.trafficlight.setPhase(tl.id, phase_index)

class Env:
    """
    Offers a similar reinforcement learning environment interface as gym.Env
    Wraps around a TrafficState (ts) and the SUMO traci (tc)
    """
    def __init__(self, c):
        self.c = c.setdefaults(redef_sumo=False, warmup_steps=0, skip_stat_steps=0, skip_vehicle_info_stat_steps=True)
        self.sumo_def = SumoDef(c)
        self.tc = None
        self.ts = None
        self.rollout_info = NamedArrays()
        self._vehicle_info = self._agent_info = None
        if c.get('vehicle_info_save'):
            self._vehicle_info = []
            if c.get('save_agent'):
                self._agent_info = []
        self._step = 0

    def def_sumo(self, *args, **kwargs):
        """ Override this with code defining the SUMO network """
        # https://sumo.dlr.de/docs/Networks/PlainXML.html
        # https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html
        return self.sumo_def.save(*args, **kwargs)

    def step(self, *args):
        """
        Override this with additional code which applies acceleration and measures observation and reward
        """
        c, ts = self.c, self.ts
        if c.get('custom_idm'):
            idm = c.custom_idm
            for veh in ts.types.human.vehicles:
                leader, headway = veh.leader()
                v = veh.speed
                s_star = 0 if leader is None else idm.minGap + max(0, v * idm.tau + v * (v - leader.speed) / (2 * np.sqrt(idm.accel * idm.decel)))
                a = idm.accel * (1 - (v / idm.maxSpeed) ** idm.delta - (s_star / (headway - leader.length)) ** 2)
                noise = np.random.normal(0, idm.sigma)
                ts.accel(veh, a + noise)

        if self._agent_info is not None:
            self.extend_agent_info()
        self.ts.step()
        self.append_step_info()
        if self._vehicle_info is not None:
            self.extend_vehicle_info()
        self._step += 1
        return c.observation_space.low, 0, False, None

    def init_vehicles(self):
        """
        Place any initial vehicles into the traffic network. Return True if the placement is successful.
        If not successful, this method will be called again.
        """
        return True

    def reset_sumo(self):
        c = self.c
        sumo_def = self.sumo_def

        generate_def = c.redef_sumo or not sumo_def.sumo_cmd
        if generate_def:
            kwargs = self.def_sumo()
            kwargs['net'] = sumo_def.generate_net(**kwargs)
            sumo_def.sumo_cmd = sumo_def.generate_sumo(**kwargs)
        self.tc = sumo_def.start_sumo(self.tc)
        if generate_def:
            self.sumo_paths = {k: p for k, p in kwargs.items() if k in SumoDef.file_args}
            defs = {k: E.from_path(p) for k, p in self.sumo_paths.items()}
            self.ts = TrafficState(c, self.tc, **defs)
        else:
            self.ts.reset(self.tc)
        self.ts.setup()
        success = self.init_vehicles()
        return success

    def init_env(self):
        ts = self.ts
        self.rollout_info = NamedArrays()
        for tl in ts.traffic_lights:
            ts.set_program(tl, 'off')
        self._step = 0
        ret = self.step()
        for _ in range(self.c.warmup_steps):
            ret = self.step()
            if ret[2] if isinstance(ret, tuple) else ret.get('done'):
                return None
        for tl in ts.traffic_lights:
            ts.set_program(tl, '1')
        if isinstance(ret, tuple):
            return ret[0]
        return {k: v for k, v in ret.items() if k in ['obs', 'id']}

    def reset(self):
        while True:
            if not self.reset_sumo():
                continue
            if (obs := self.init_env()) is not None:
                return obs

    def append_step_info(self):
        """
        Save simulation statistics at the current step
        """
        ts = self.ts
        rl, human = ts.types.rl, ts.types.human
        self.rollout_info.append(
            id=ts.vehicles.keys(),
            speed=[veh.speed for veh in ts.vehicles],
            speed_rl=[veh.speed for veh in rl.vehicles],
            speed_human=[veh.speed for veh in human.vehicles],
            fuel=[veh.fuelconsumption for veh in ts.vehicles],
            collisions=len(ts.new_collided),
            collisions_human=len([veh for veh in ts.new_collided if veh.type is human]),
            inflow=len(ts.new_departed),
            outflow=len(ts.new_arrived),
            backlog=sum(len(f.backlog) for f in ts.flows)
        )

    def extend_vehicle_info(self):
        """
        Save vehicle state at the current step
        """
        step = self._step
        c = self.c
        if c.skip_vehicle_info_stat_steps and step < c.warmup_steps + c.skip_stat_steps:
            return
        self._vehicle_info.extend(
            (step, veh.id, veh.type.id, veh.speed, veh.lane.id, veh.laneposition, veh.fuelconsumption)
        for veh in self.ts.vehicles)

    def extend_agent_info(self):
        """
        Save agent state at the current step
        """
        step = self._step
        c = self.c
        if c.skip_vehicle_info_stat_steps and step < c.warmup_steps + c.skip_stat_steps:
            return
        self._agent_info.extend(
            (step, veh.id, veh.get('accel', None), veh.get('lane_change', None))
        for veh in self.ts.types.rl.vehicles)

    @property
    def stats(self):
        """
        Get all the previously saved stats
        """
        c = self.c
        info = self.rollout_info[1 + c.warmup_steps + c.skip_stat_steps:] # + 1 is for the first step used to set up reset()
        mean = lambda L: np.mean(L) if len(L) else np.nan
        std = lambda L: np.std(L) if len(L) else np.nan
        unique = np.unique(flatten(info.id))
        return Namespace(
            horizon=len(info),
            speed=mean(flatten(info.speed)),
            speed_rl=mean(flatten(info.speed_rl)),
            speed_rl_std=std(flatten(info.speed_rl)),
            speed_human=mean(flatten(info.speed_human)),
            inflow=sum(info.inflow),
            outflow=sum(info.outflow),
            inflow_hourly=sum(info.inflow) * 3600 / ((len(info.inflow) or np.nan) * c.sim_step),
            outflow_hourly=sum(info.outflow) * 3600 / ((len(info.outflow) or np.nan) * c.sim_step),
            collisions=sum(info.collisions),
            collisions_human=sum(info.collisions_human),
            fuel=sum(flatten(info.fuel)) / (len(unique) or np.nan),
        )

    @property
    def vehicle_info(self):
        """
        Get all the previously saved vehicle states
        """
        return pd.DataFrame(self._vehicle_info, columns=['step', 'id', 'type', 'speed', 'lane_id', 'lane_position', 'fuel'])

    @property
    def agent_info(self):
        """
        Get all the previously saved agent actions
        """
        return pd.DataFrame(self._agent_info, columns=['step', 'id', 'accel', 'lane_change'])

    def close(self):
        try: traci.close()
        except: pass

class NormEnv(gym.Env):
    """
    Reward normalization with running average https://github.com/joschu/modular_rl
    """
    def __init__(self, c, env):
        self.c = c.setdefaults(norm_obs=False, norm_reward=False, center_reward=False, reward_clip=np.inf, obs_clip=np.inf)
        self.env = env
        self._obs_stats = RunningStats()
        self._return_stats = RunningStats()
        self._reward_stats = RunningStats()
        self._running_ret = 0

    def norm_obs(self, obs):
        c = self.c
        if c.norm_obs:
            self._obs_stats.update(obs)
            obs = (obs - self._obs_stats.mean) / (self._obs_stats.std + 1e-8)
            obs = np.clip(obs, -c.obs_clip, c.obs_clip)
        return obs

    def norm_reward(self, reward):
        c = self.c
        if c.center_reward:
            self._reward_stats.update(reward)
            reward = reward - self._reward_stats.mean
        if c.norm_reward:
            self._running_ret = self._running_ret * c.gamma + reward # estimation of return
            self._return_stats.update(self._running_ret)
            normed_r = reward / (self._return_stats.std + 1e-8) # norm reward by std of return
            reward = np.clip(normed_r, -c.reward_clip, c.reward_clip)
        return reward

    def reset(self):
        return self.norm_obs(self.env.reset())

    def step(self, action=None):
        ret = self.env.step(action)
        if isinstance(ret, tuple):
            obs, reward, done, info = ret
            norm_obs = self.norm_obs(obs)
            norm_rew = self.norm_reward(reward)
            return norm_obs, norm_rew, done, reward
        else:
            if 'obs' in ret:
                ret['obs'] = self.norm_obs(ret['obs'])
            ret['raw_reward'] = ret['reward']
            ret['reward'] = self.norm_reward(ret['reward'])
            return ret

    def __getattr__(self, attr):
        try:
            return self.__getattribute__(attr)
        except:
            return getattr(self.env, attr)
