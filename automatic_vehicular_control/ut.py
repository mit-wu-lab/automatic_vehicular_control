import scipy.signal
from gym.spaces import Box, Discrete

from automatic_vehicular_control.u import *

def discount(x, gamma):
    if isinstance(x, torch.Tensor):
        n = x.size(0)
        return F.conv1d(F.pad(x, (0, n - 1)).view(1, 1, -1), gamma ** torch.arange(n, dtype=x.dtype).view(1, 1, -1)).view(-1)
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def explained_variance(y_pred, y_true):
    if not len(y_pred):
        return np.nan
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

class RunningStats:
    '''
    Tracks first and second moments (mean and variance) of a streaming time series
    https://github.com/joschu/modular_rl
    http://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self, stats={}):
        self.n = self.mean = self._nstd = 0
        self.__dict__.update(stats)

    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean
            self.mean = old_mean + (x - old_mean) / self.n
            self._nstd = self._nstd + (x - old_mean) * (x - self.mean)
    @property
    def var(self):
        return self._nstd / (self.n - 1) if self.n > 1 else np.square(self.mean)
    @property
    def std(self):
        return np.sqrt(self.var)

class NamedArrays(dict):
    """
    Data structure for keeping track of a dictionary of arrays (used for rollout information)
    e.g. {'reward': [...], 'action': [...]}
    """
    def __init__(self, dict_of_arrays={}, **kwargs):
        kwargs.update(dict_of_arrays)
        super().__init__(kwargs)

    def __getattr__(self, k):
        if k in self.__dict__:
            return self.__dict__[k]
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v

    def __getitem__(self, k):
        if isinstance(k, (slice, int, np.ndarray, list)):
            return type(self)((k_, arr[k]) for k_, arr in self.items())
        return super().__getitem__(k)

    def __setitem__(self, k, v):
        if isinstance(k, (slice, int, np.ndarray, list)):
            if isinstance(v, dict):
                for vk, vv in v.items():
                    self[vk][k] = vv
            else:
                for k_, arr in self.items():
                    arr[k] = v
        else:
            super().__setitem__(k, v)

    def append(self, *args, **kwargs):
        for k, v in itertools.chain(args, kwargs.items()):
            if isinstance(v, dict):
                self.setdefault(k, type(self)()).append(**v)
            else:
                self.setdefault(k, []).append(v)

    def extend(self, *args, **kwargs):
        for k, v in itertools.chain(args, kwargs.items()):
            if isinstance(v, dict):
                self.setdefault(k, type(self)()).extend(**v)
            else:
                self.setdefault(k, []).extend(v)

    def to_array(self, inplace=True, dtype=None, concat=False):
        return self.apply(np.concatenate if concat else lambda x: np.asarray(x, dtype=dtype), inplace)

    def to_torch(self, dtype=None, device=None):
        for k, v in self.items():
            if isinstance(v, list):
                v = np.asarray(v, dtype=dtype)
            if isinstance(v, NamedArrays):
                v.to_torch(dtype, device)
            else:
                if v.dtype == np.object:
                    self[k] = [torch.tensor(np.ascontiguousarray(x), device=device) for x in v]
                else:
                    self[k] = torch.tensor(np.ascontiguousarray(v), device=device)
        return self

    def trim(self):
        min_len = len(self)
        for k, v in self.items():
            self[k] = v[:min_len]

    def __len__(self):
        return len(self.keys()) and min(len(v) for v in self.values())

    def filter(self, *args):
        return type(self)((k, v) for k, v in self.items() if k in args)

    def iter_minibatch(self, n_minibatches=None, concat=False, device='cpu'):
        if n_minibatches in [1, None]:
            yield slice(None), self.to_array(inplace=False, concat=concat).to_torch(device=device)
        else:
            for idxs in np.array_split(np.random.permutation(len(self)), n_minibatches):
                na = type(self)((k, v[idxs]) for k, v in self.items())
                yield idxs, na.to_array(inplace=False, concat=concat).to_torch(device=device)

    def apply(self, fn, inplace=True):
        if inplace:
            for k, v in self.items():
                if isinstance(v, NamedArrays):
                    v.apply(fn)
                else:
                    self[k] = fn(v)
            return self
        else:
            return type(self)((k, v.apply(fn, inplace=False) if isinstance(v, NamedArrays) else fn(v)) for k, v in self.items())

    def __getstate__(self):
        return dict(**self)

    def __setstate__(self, d):
        self.update(d)

    @classmethod
    def concat(cls, named_arrays, fn=None):
        named_arrays = list(named_arrays)
        if not len(named_arrays):
            return cls()
        def concat(xs):
            """
            Common error with np.concatenate: conside arrays a and b, both of which are lists of arrays. If a contains irregularly shaped arrays and b contains arrays with the same shape, the numpy will treat b as a 2D array, and the concatenation will fail. Solution: use flatten instead of np.concatenate for lists of arrays
            """
            try: return np.concatenate(xs)
            except: return flatten(xs)
        get_concat = lambda v: v.concat if isinstance(v, NamedArrays) else fn or (torch.cat if isinstance(v, torch.Tensor) else concat)
        return cls((k, get_concat(v)([x[k] for x in named_arrays])) for k, v in named_arrays[0].items())

class Dist:
    """ Distribution interface """
    def __init__(self, inputs):
        self.inputs = inputs

    def sample(self, shape=torch.Size([])):
        return self.dist.sample(shape)

    def argmax(self):
        raise NotImplementedError

    def logp(self, actions):
        return self.dist.log_prob(actions)

    def kl(self, other):
        return torch.distributions.kl.kl_divergence(self.dist, other.dist)

    def entropy(self):
        return self.dist.entropy()

    def __getitem__(self, idx):
        return type(self)(self.inputs[idx])

class CatDist(Dist):
    """ Categorical distribution (for discrete action spaces) """
    def __init__(self, inputs):
        super().__init__(inputs)
        self.dist = torch.distributions.categorical.Categorical(logits=inputs, validate_args=False)

    def argmax(self):
        return self.dist.probs.argmax(dim=-1)

class DiagGaussianDist(Dist):
    """ Diagonal Gaussian distribution (for continuous action spaces) """
    def __init__(self, inputs):
        super().__init__(inputs)
        self.mean, self.log_std = torch.chunk(inputs, 2, dim=-1)
        self.std = self.log_std.exp()
        self.dist = torch.distributions.normal.Normal(self.mean, self.std)

    def argmax(self):
        return self.dist.mean

    def logp(self, actions):
        return super().logp(actions).sum(dim=-1)

    def kl(self, other):
        return super().kl(other).squeeze(dim=-1)

    def entropy(self):
        return super().entropy().squeeze(dim=-1)

def build_dist(space):
    """
    Build a nested distribution
    """
    if isinstance(space, Box):
        class DiagGaussianDist_(DiagGaussianDist):
            model_output_size = np.prod(space.shape) * 2
        return DiagGaussianDist_
    elif isinstance(space, Discrete):
        class CatDist_(CatDist):
            model_output_size = space.n
        return CatDist_

    assert isinstance(space, dict) # Doesn't support lists at the moment since there's no list equivalent of NamedArrays that allows advanced indexing
    names, subspaces = zip(*space.items())
    to_list = lambda x: [x[name] for name in names]
    from_list = lambda x: NamedArrays(zip(names, x))
    subdist_classes = [build_dist(subspace) for subspace in subspaces]
    subsizes = [s.model_output_size for s in subdist_classes]
    class Dist_(Dist):
        model_output_size = sum(subsizes)
        def __init__(self, inputs):
            super().__init__(inputs)
            self.dists = from_list(cl(x) for cl, x in zip(subdist_classes, inputs.split(subsizes, dim=-1)))

        def sample(self, shape=torch.Size([])):
            return from_list([dist.sample(shape) for dist in to_list(self.dists)])

        def argmax(self):
            return from_list([dist.argmax() for dist in to_list(self.dists)])

        def logp(self, actions):
            return sum(dist.logp(a) for a, dist in zip(to_list(actions), to_list(self.dists)))

        def kl(self, other):
            return sum(s.kl(o) for s, o in zip(to_list(self.dists), to_list(other.dists)))

        def entropy(self):
            return sum(dist.entropy() for dist in to_list(self.dists))
    return Dist_

def build_fc(input_size, *sizes_and_modules):
    """
    Build a fully connected network
    """
    layers = []
    str_map = dict(relu=nn.ReLU(inplace=True), tanh=nn.Tanh(), sigmoid=nn.Sigmoid(), flatten=nn.Flatten(), softmax=nn.Softmax())
    for x in sizes_and_modules:
        if isinstance(x, (int, np.integer)):
            input_size, x = x, nn.Linear(input_size, x)
        if isinstance(x, str):
            x = str_map[x]
        layers.append(x)
    return nn.Sequential(*layers)

class FFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c.setdefaults(layers=[64, 'tanh', 64, 'tanh'], weight_scale='default', weight_init='orthogonal')
        layers = c.layers
        if isinstance(layers, list):
            layers = Namespace(s=[], v=layers, p=layers)
        s_sizes = [c.observation_space.shape[0], *layers.s]

        self.shared = build_fc(*s_sizes)

        self.p_head = build_fc(s_sizes[-1], *layers.p, c.model_output_size)
        self.sequential_init(self.p_head, 'policy')
        self.v_head = None
        if c.use_critic:
            self.v_head = build_fc(s_sizes[-1], *layers.v, 1)
            self.sequential_init(self.v_head, 'value')

    def sequential_init(self, seq, key):
        c = self.c
        linears = [m for m in seq if isinstance(m, nn.Linear)]
        for i, m in enumerate(linears):
            if isinstance(c.weight_scale, (int, float)):
                scale = c.weight_scale
            elif isinstance(c.weight_scale, (list, tuple)):
                scale = c.weight_scale[i]
            elif isinstance(c.weight_scale, (dict)):
                scale = c.weight_scale[key][i]
            else:
                scale = 0.01 if m == linears[-1] else 1
            if c.weight_init == 'normc': # normalize along input dimension
                weight = torch.randn_like(m.weight)
                m.weight.data = weight * scale / weight.norm(dim=1, keepdim=True)
            elif c.weight_init == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=scale)
            elif c.weight_init == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=scale)
            nn.init.zeros_(m.bias)

    def forward(self, inp, value=False, policy=False, argmax=None):
        s = self.shared(inp)
        pred = Namespace()
        if value and self.v_head:
            pred.value = self.v_head(s).view(-1)
        if policy or argmax is not None:
            pred.policy = self.p_head(s)
            if argmax is not None:
                dist = self.c.dist_class(pred.policy)
                pred.action = dist.argmax() if argmax else dist.sample()
        return pred

def calc_adv(reward, gamma, value_=None, lam=None):
    """
    Calculate advantage with TD-lambda
    """
    if value_ is None:
        return discount(reward, gamma), None # TD(1)
    if isinstance(reward, list):
        reward, value_ = map(np.array, (reward, value_))
    assert value_.ndim == reward.ndim == 1, f'Value and reward be one dimensional, but got {value_.shape} and {reward.shape} respectively'
    assert value_.shape[0] - reward.shape[0] in [0, 1], f'Value\'s shape can be at most 1 bigger than reward\'s shape, but got {value_.shape} and {reward.shape} respectively'

    if value_.shape[0] == reward.shape[0]:
        delta = reward - value_
        delta[:-1] += gamma * value_[1:]
    else:
        delta = reward + gamma * value_[1:] - value_[:-1]
    adv = discount(delta, gamma * lam)
    ret = value_[:len(adv)] + adv
    return ret, adv

def calc_adv_multi_agent(id_, reward, gamma, value_=None, lam=None):
    """
    Calculate advantage with TD-lambda for multiple agents

    id_ and value_ include the last time step, reward does not include the last time step
    id_ should be something that pandas.Series.groupby works on
    id_, reward, and value_ should be flat arrays with "shape" n_steps * n_agent_per_step
    """
    n_id = len(reward) # number of ids BEFORE the last time step
    ret = np.empty((n_id,), dtype=np.float32)
    adv = ret.copy()
    for _, group in pd.Series(id_).groupby(id_):
        idxs = group.index
        value_i_ = None if value_ is None else value_[idxs]
        if idxs[-1] >= n_id:
            idxs = idxs[:-1]
        ret[idxs], adv[idxs] = calc_adv(reward=reward[idxs], gamma=gamma, value_=value_i_, lam=lam)
    return ret, adv

class Algorithm:
    """
    RL algorithm interface
    """
    def __init__(self, c):
        self.c = c.setdefaults(normclip=None, use_critic=True, lam=1, batch_concat=False, device='cpu')

    def on_step_start(self):
        return {}

    def optimize(self, rollouts):
        raise NotImplementedError

    def value_loss(self, v_pred, ret, v_start=None, mask=None):
        c = self.c
        mask = slice(None) if mask is None else mask
        unclipped = (v_pred - ret) ** 2
        if v_start is None or c.vclip is None: # no value clipping
            return unclipped[mask].mean()
        clipped_value = v_start + (v_pred - v_start).clamp(-c.vclip, c.vclip)
        clipped = (clipped_value - ret) ** 2
        return torch.max(unclipped, clipped)[mask].mean() # no gradient if larger

    def step_loss(self, loss):
        c = self.c
        c._opt.zero_grad()
        if torch.isnan(loss):
            import q; q.d()
            raise RuntimeError('Encountered nan loss during training')
        loss.backward()
        if c.normclip:
            torch.nn.utils.clip_grad_norm_(c._model.parameters(), c.normclip)
        c._opt.step()

class ValueFitting(Algorithm):
    def __init__(self, c):
        super().__init__(c.setdefaults(vclip=None))

    def optimize(self, rollouts):
        c = self.c
        buffer = c._buffer[:c._buffer_size]
        for i_gd in range(c.n_gds):
            batch_stats = []
            for idxs, mb in buffer.iter_minibatch(max(1, c._buffer_size // c.size_mb), concat=c.batch_concat, device=c.device):
                pred = c._model(mb.obs, value=True)
                value_mask = mb.obs.get('value_mask') if isinstance(mb.obs, dict) else None
                loss = self.value_loss(pred.value, mb.ret, v_start=mb.value, mask=value_mask)
                self.step_loss(loss)
                batch_stats.append(from_torch(dict(loss=loss)))
            c.log_stats(pd.DataFrame(batch_stats).mean(axis=0), ii=i_gd, n_ii=c.n_gds)
        c.flush_writer_buffer()

class Imitation(Algorithm):
    def __init__(self, c):
        super().__init__(c.setdefaults(n_gds=10, lr=1e-3, vcoef=1, vclip=None))

    def optimize(self, rollouts):
        c = self.c
        buffer = c._buffer[:c._buffer_size]

        for i_gd in range(c.n_gds):
            batch_stats = NamedArrays()
            for _, mb in buffer.iter_minibatch(max(1, c._buffer_size // c.size_mb), concat=c.batch_concat, device=c.device):
                pred = c._model(mb.obs, policy=True, value=True)
                curr_dist = c.dist_class(pred.policy)
                label = mb.label
                n = len(label)
                n_pos = label.sum()
                n_neg = n - n_pos
                if isinstance(c.action_space, Box):
                    z = torch.normal(0, 1, (n, 10), device=c.device) # pathwise estimator
                    samples = curr_dist.mean + curr_dist.std * z
                    samples = (samples.clamp(c.low, 1) - c.low) / (1 - c.low)
                    mse = ((label.view(-1, 1) - samples) ** 2).mean(dim=1)
                    entropy = curr_dist.entropy().mean()
                    ent_loss = -c.entcoef * entropy
                    if c.get('imitation_balance'):
                        pos_loss = label.dot(mse) / n_pos if n_pos > 0 else 0
                        neg_loss = (1 - label).dot(mse) / n_neg if n_neg > 0 else 0
                        policy_loss = pos_loss + neg_loss + ent_loss
                        batch_stats.append(policy_loss=policy_loss,n_pos=n_pos, pos_loss=pos_loss, n_neg=n_neg, neg_loss=neg_loss, entropy=entropy, ent_loss=ent_loss, std=curr_dist.std.mean())
                    else:
                        mse_loss = mse.mean()
                        policy_loss = mse_loss + ent_loss
                        batch_stats.append(policy_loss=policy_loss, mse_loss=mse_loss, entropy=entropy, ent_loss=ent_loss, std=curr_dist.std.mean())
                else:
                    logp = curr_dist.logp(mb.label.long())
                    if c.get('imitation_balance'):
                        pos_loss = -label.dot(logp) / n_pos
                        neg_loss = -(1 - label).dot(logp) / n_neg
                        policy_loss = pos_loss + neg_loss
                        batch_stats.append(policy_loss=policy_loss, pos_loss=pos_loss, neg_loss=neg_loss)
                    else:
                        policy_loss = logp.mean()
                        batch_stats.append(policy_loss=policy_loss)
                if c.use_critic:
                    value_loss = self.value_loss(pred.value, mb.ret, v_start=mb.value, mask=mb.obs.get('value_mask'))
                    batch_stats.append(value_loss=value_loss)
                    loss = policy_loss + value_loss * c.vcoef
                else:
                    loss = policy_loss
                self.step_loss(loss)
            c.log_stats(dict(ii=i_gd, n_ii=c.n_gds, **pd.DataFrame(from_torch(batch_stats)).mean(axis=0)))
        c.flush_writer_buffer()

class PPO(Algorithm):
    def __init__(self, c):
        super().__init__(c.setdefaults(use_critic=True, n_gds=30, pclip=0.3, vcoef=1, vclip=1, klcoef=0.2, kltarg=0.01, entcoef=0))

    def on_step_start(self):
        stats = dict(klcoef=self.c.klcoef)
        if self.c.entcoef:
            stats['entcoef'] = self.entcoef
        return stats

    @property
    def entcoef(self):
        c = self.c
        return c.schedule(c.entcoef, c.get('ent_schedule'))

    def optimize(self, rollouts):
        c = self.c
        batch = rollouts.filter('obs', 'policy', 'action', 'pg_obj', 'ret', *lif(c.use_critic, 'value', 'adv'))
        value_warmup = c._i < c.get('n_value_warmup', 0)

        for i_gd in range(c.n_gds):
            batch_stats = []
            for idxs, mb in batch.iter_minibatch(c.get('n_minibatches'), concat=c.batch_concat, device=c.device):
                if not len(mb):
                    continue
                start_dist = c.dist_class(mb.policy)
                start_logp = start_dist.logp(mb.action)
                if 'pg_obj' not in batch:
                    mb.pg_obj = mb.adv if c.use_critic else mb.ret
                pred = c._model(mb.obs, value=True, policy=True)
                curr_dist = c.dist_class(pred.policy)
                p_ratio = (curr_dist.logp(mb.action) - start_logp).exp()

                pg_obj = mb.pg_obj
                if c.adv_norm:
                    pg_obj = normalize(pg_obj)

                policy_loss = -torch.min(
                    pg_obj * p_ratio,
                    pg_obj * p_ratio.clamp(1 - c.pclip, 1 + c.pclip) # no gradient if larger
                ).mean()

                kl = start_dist.kl(curr_dist).mean()
                entropy = curr_dist.entropy().mean()

                loss = policy_loss + c.klcoef * kl - self.entcoef * entropy
                stats = dict(
                    policy_loss=policy_loss, kl=kl, entropy=entropy
                )

                if value_warmup:
                    loss = loss.detach()

                if c.use_critic:
                    value_mask = mb.obs.get('value_mask') if isinstance(mb.obs, dict) else None
                    value_loss = self.value_loss(pred.value, mb.ret, v_start=mb.value, mask=value_mask)
                    loss += c.vcoef * value_loss
                    stats['value_loss'] = value_loss
                self.step_loss(loss)
                batch_stats.append(from_torch(stats))
            c.log_stats(pd.DataFrame(batch_stats).mean(axis=0), ii=i_gd, n_ii=c.n_gds)
        c.flush_writer_buffer()

        if c.klcoef:
            kl = from_torch(kl)
            if kl > 2 * c.kltarg:
                c.klcoef *= 1.5
            elif kl < 0.5 * c.kltarg:
                c.klcoef *= 0.5

class PG(PPO):
    def __init__(self, c):
        super().__init__(c.var(n_gds=1, vcoef=0, klcoef=0, kltarg=0, use_critic=False))

class TRPO(Algorithm):
    def __init__(self, c):
        if c.get('max_kl'):
            c.start_max_kl = c.end_max_kl = c.max_kl
        super().__init__(c.setdefaults(
            use_critic=True, start_max_kl=0.01, end_max_kl=0.01, steps_cg=10, steps_backtrack=10, damping=0.1, accept_ratio=0.1, n_gds=1))

    @property
    def max_kl(self):
        c = self.c
        return c.start_max_kl + (c.end_max_kl - c.start_max_kl) * c._i / c.n_steps

    def on_step_start(self):
        return dict(max_kl=self.max_kl)

    def optimize(self, rollouts):
        c = self.c
        batch = rollouts.filter('obs', 'policy', 'action', 'pg_obj', 'ret', *lif(c.use_critic, 'value', 'adv'))
        (_, b), = batch.iter_minibatch(None, concat=c.batch_concat, device=c.device) # to be consistent with PPO
        pg_obj = b.pg_obj if 'pg_obj' in b else b.adv if c.use_critic else b.ret

        if c.adv_norm:
            pg_obj = normalize(pg_obj)
        start_dist = c.dist_class(b.policy)
        start_logp = start_dist.logp(b.action)
        def surrogate(dist):
            p_ratio = (dist.logp(b.action) - start_logp).exp()
            return (pg_obj * p_ratio).mean()

        # objective is -policy_loss, actually here the p_ratio is just 1, but we care about the gradients
        pred = c._model(b.obs, value=True, policy=True)
        pred_dist = c.dist_class(pred.policy)
        obj = surrogate(pred_dist)

        from torch.nn.utils import parameters_to_vector, vector_to_parameters

        params = list(c._model.p_head.parameters())
        flat_start_params = parameters_to_vector(params).clone()
        grad_obj = parameters_to_vector(torch.autograd.grad(obj, params, retain_graph=True))

        # Make fisher product estimator
        kl = start_dist.kl(pred_dist).mean() # kl is 0, but we care about the gradient
        grad_kl = parameters_to_vector(torch.autograd.grad(kl, params, create_graph=True))

        def fvp_fn(x): # fisher vector product
            return parameters_to_vector(
                torch.autograd.grad(grad_kl @ x, params, retain_graph=True)
            ).detach() + x * c.damping

        def cg_solve(fvp_fn, b, nsteps):
            """
            Conjugate Gradients Algorithm
            Solves Hx = b, where H is the Fisher matrix and b is known
            """
            x = torch.zeros_like(b) # solution
            r = b.clone() # residual
            p = b.clone() # direction
            new_rnorm = r @ r
            for _ in range(nsteps):
                rnorm = new_rnorm
                fvp = fvp_fn(p)
                alpha = rnorm / (p @ fvp)
                x += alpha * p
                r -= alpha * fvp
                new_rnorm = r @ r
                p = r + new_rnorm / rnorm * p
            return x
        step = cg_solve(fvp_fn, grad_obj, c.steps_cg)
        max_trpo_step = (2 * self.max_kl / (step @ fvp_fn(step))).sqrt() * step

        with torch.no_grad(): # backtrack until we find best step
            improve_thresh = grad_obj @ max_trpo_step * c.accept_ratio
            step = max_trpo_step
            for i_scale in range(c.steps_backtrack):
                vector_to_parameters(flat_start_params + step, params)
                test_dist = c.dist_class(c._model(b.obs, policy=True).policy)
                test_obj = surrogate(test_dist)
                kl = start_dist.kl(test_dist).mean()
                if kl < self.max_kl and test_obj - obj > improve_thresh:
                    break
                step /= 2
                improve_thresh /= 2
            else:
                vector_to_parameters(flat_start_params, params)
                test_dist = start_dist
                test_obj = obj
                kl = 0

        if c.use_critic:
            shared = getattr(c._model, 'shared', None)
            if shared is not None:
                assert len(shared) == 0, 'Value network and policy network cannot share weights'
            for i_gd in range(c.n_gds):
                batch_stats = []
                for idxs, mb in rollouts.iter_minibatch(c.get('n_minibatches'), concat=c.batch_concat, device=c.device):
                    pred = c._model(mb.obs, value=True)
                    value_mask = mb.obs.get('value_mask') if isinstance(mb.obs, dict) else None
                    value_loss = self.value_loss(pred.value, mb.ret, mask=value_mask)
                    self.step_loss(value_loss)
                    batch_stats.append(dict(value_loss=from_torch(value_loss)))
                c.log_stats(pd.DataFrame(batch_stats).mean(axis=0), ii=i_gd, n_ii=c.n_gds)
        c.flush_writer_buffer()
        entropy = test_dist.entropy().mean()
        c.log_stats(from_torch(dict(policy_loss=-obj, final_policy_loss=-test_obj, i_scale=i_scale, kl=kl, entropy=entropy)))
