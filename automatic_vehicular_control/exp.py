from automatic_vehicular_control.u import Config
from automatic_vehicular_control.ut import *

class Main(Config):
    flow_base = Path.env('F')._real

    never_save = {'trial', 'has_av', 'e', 'disable_amp', 'opt_level', 'tmp'} | Config.never_save

    def __init__(c, res, *args, **kwargs):
        tmp = Path(res)._real in [Path.env('HOME'), Main.flow_base, Main.flow_base / 'all_results']
        if tmp:
            res = Main.flow_base / 'tmp' / rand_string(8)
        kwargs.setdefault('disable_amp', True)
        kwargs.setdefault('tmp', tmp)
        super().__init__(res, *args, **kwargs)
        if tmp:
            os.environ['WANDB_MODE'] = 'dryrun'
        c.setdefaults(e=False, tb=True, wb=False)
        c.logger = c.logger and c.e is False
        if tmp and c.e is False:
            res.mk()
            c.log('Temporary run for testing with res=%s' % res)

    def create_env(c):
        raise NotImplementedError

    @property
    def dist_class(c):
        if '_dist_class' not in c:
            c._dist_class = build_dist(c.action_space)
        return c._dist_class

    @property
    def model_output_size(c):
        if '_model_output_size' not in c:
            c._model_output_size = c.dist_class.model_output_size
        return c._model_output_size

    @property
    def observation_space(c):
        raise NotImplementedError

    @property
    def action_space(c):
        raise NotImplementedError

    def set_model(c):
        c._model = c.get('model_cls', FFN)(c)
        return c

    def schedule(c, coef, schedule=None):
        if not schedule and isinstance(coef, (float, int)):
            return coef
        frac = c._i / c.n_steps
        frac_left = 1 - frac
        if callable(coef):
            return coef(frac_left)
        elif schedule == 'linear':
            return coef * frac_left
        elif schedule == 'cosine':
            return coef * (np.cos(frac * np.pi) + 1) / 2

    @property
    def _lr(c):
        return c.schedule(c.get('lr', 1e-4), c.get('lr_schedule'))

    def log_stats(c, stats, ii=None, n_ii=None, print_time=False):
        stats = {k: v for k, v in stats.items() if v is not None}
        total_time = time() - c._run_start_time
        if print_time:
            stats['total_time'] = total_time

        prints = []
        if ii is not None:
            prints.append('ii {:2d}'.format(ii))

        prints.extend('{} {:.3g}'.format(*kv) for kv in stats.items())

        widths = [len(x) for x in prints]
        line_w = terminal_width()
        prefix = 'i {:d}'.format(c._i)
        i_start = 0
        curr_w = len(prefix) + 3
        curr_prefix = prefix
        for i, w in enumerate(widths):
            if curr_w + w > line_w:
                c.log(' | '.join([curr_prefix, *prints[i_start: i]]))
                i_start = i
                curr_w = len(prefix) + 3
                curr_prefix = ' ' * len(prefix)
            curr_w += w + 3
        c.log(' | '.join([curr_prefix, *prints[i_start:]]))
        sys.stdout.flush()

        if ii is not None:
            c._writer_buffer.append(**stats)
            return

        c.add_stats(stats)

    def flush_writer_buffer(c):
        if len(c._writer_buffer):
            stats = {k: np.nanmean(c._writer_buffer.pop(k)) for k in list(c._writer_buffer)}
            c.add_stats(stats)

    def add_stats(c, stats):
        total_time = time() - c._run_start_time
        df = c._results
        for k, v in stats.items():
            if k not in df:
                df[k] = np.nan
            df.loc[c._i, k] = v

        if c.e is False:
            if c.tb:
                for k, v in stats.items():
                    c._writer.add_scalar(k, v, global_step=c._i, walltime=total_time)
            if c.wb:
                c._writer.log(stats, step=c._i)

    def get_log_ii(c, ii, n_ii=None, print_time=False):
        return lambda **kwargs: c.log_stats(kwargs, ii, print_time=print_time)

    def on_rollout_worker_start(c):
        c._env = c.create_env()
        c.use_critic = False # Don't need value function on workers
        c.set_model()
        c._model.eval()
        c._i = 0

    def set_weights(c, weights): # For Ray
        c._model.load_state_dict(weights, strict=False) # If c.use_critic, worker may not have critic weights

    def on_train_start(c):
        c.setdefaults(alg='Algorithm')
        c._env = c.create_env()

        c._alg = (eval(c.alg) if isinstance(c.alg, str) else c.alg)(c)
        c.set_model()
        c._model.train()
        c._model.to(c.device)

        c._i = 0 # for c._lr
        opt = c.get('opt', 'Adam')
        if opt == 'Adam':
            c._opt = optim.Adam(c._model.parameters(), lr=c._lr, betas=c.get('betas', (0.9, 0.999)), weight_decay=c.get('l2', 0))
        elif opt == 'RMSprop':
            c._opt = optim.RMSprop(c._model.parameters(), lr=c._lr, weight_decay=c.get('l2', 0))

        c._run_start_time = time()
        c._i = c.set_state(c._model, opt=c._opt, step='max')
        if c._i:
            c._results = c.load_train_results().loc[:c._i]
            c._run_start_time -= c._results.loc[c._i, 'total_time']
        else:
            c._results = pd.DataFrame(index=pd.Series(name='step'))
        c._i_gd = None

        c.try_save_commit(Main.flow_base)

        if c.tb:
            from torch.utils.tensorboard import SummaryWriter
            c._writer = SummaryWriter(log_dir=c.res, flush_secs=10)
        if c.wb:
            import wandb
            wandb_id_path = (c.res / 'wandb' / 'id.txt').dir_mk()
            c._wandb_run = wandb.init( # name and project should be set as env vars
                name=c.res.rel(Path.env('FA')),
                dir=c.res,
                id=wandb_id_path.load() if wandb_id_path.exists() else None,
                config={k: v for k, v in c.items() if not k.startswith('_')},
                save_code=False
            )
            wandb_id_path.save(c._wandb_run.id)
            c._writer = wandb
        c._writer_buffer = NamedArrays()

    def on_step_start(c, stats={}):
        lr = c._lr
        for g in c._opt.param_groups:
            g['lr'] = float(lr)
        c.log_stats(dict(**stats, **c._alg.on_step_start(), lr=lr))

        if c._i % c.step_save == 0:
            c.save_train_results(c._results)
            c.save_state(c._i, c.get_state(c._model, c._opt, c._i))

    def rollouts(c):
        """ Collect a list of rollouts for the training step """
        if c.use_ray:
            import ray
            weights_id = ray.put({k: v.cpu() for k, v in c._model.state_dict().items()})
            [w.set_weights.remote(weights_id) for w in c._rollout_workers]
            rollout_stats = flatten(ray.get([w.rollouts_single_process.remote() for w in c._rollout_workers]))
        else:
            rollout_stats = c.rollouts_single_process()
        rollouts = [c.on_rollout_end(*rollout_stat, ii=ii) for ii, rollout_stat in enumerate(rollout_stats)]
        c.flush_writer_buffer()
        return NamedArrays.concat(rollouts, fn=flatten)

    def rollouts_single_process(c):
        if c.n_rollouts_per_worker > 1:
            rollout_stats = [c.var(i_rollout=i).rollout() for i in range(c.n_rollouts_per_worker)]
        else:
            n_steps_total = 0
            rollout_stats = []
            while n_steps_total < c.horizon:
                if c.get('full_rollout_only'):
                    n_steps_total = 0
                    rollout_stats = []
                rollout, stats = c.rollout()
                rollout_stats.append((rollout, stats))
                n_steps_total += stats.get('horizon') or len(stats.get('reward', []))
        return rollout_stats

    def get_env_stats(c):
        return c._env.stats

    def rollout(c):
        c.setdefaults(skip_stat_steps=0, i_rollout=0, rollout_kwargs=None)
        if c.rollout_kwargs and c.e is False:
            c.update(c.rollout_kwargs[c.i_rollout])
        t_start = time()

        ret = c._env.reset()
        if not isinstance(ret, dict):
            ret = dict(obs=ret)
        rollout = NamedArrays()
        rollout.append(**ret)

        done = False
        a_space = c.action_space
        step = 0
        while step < c.horizon + c.skip_stat_steps and not done:
            pred = from_torch(c._model(to_torch(rollout.obs[-1]), value=False, policy=True, argmax=False))
            if c.get('aclip', True) and isinstance(a_space, Box):
                pred.action = np.clip(pred.action, a_space.low, a_space.high)
            rollout.append(**pred)

            ret = c._env.step(rollout.action[-1])
            if isinstance(ret, tuple):
                obs, reward, done, info = ret
                ret = dict(obs=obs, reward=reward, done=done, info=info)
            done = ret.setdefault('done', False)
            if done:
                ret = {k: v for k, v in ret.items() if k not in ['obs', 'id']}
            rollout.append(**ret)
            step += 1
        stats = dict(rollout_time=time() - t_start, **c.get_env_stats())
        return rollout, stats

    def on_rollout_end(c, rollout, stats, ii=None):
        """ Compute value, calculate advantage, log stats """
        t_start = time()
        step_id_ = rollout.pop('id', None)
        done = rollout.pop('done', None)
        multi_agent = step_id_ is not None

        step_obs_ = rollout.obs
        step_obs = step_obs_ if done[-1] else step_obs_[:-1]
        assert len(step_obs) == len(rollout.reward)

        value_ = None
        if c.use_critic:
            (_, mb_), = rollout.filter('obs').iter_minibatch(concat=multi_agent, device=c.device)
            value_ = from_torch(c._model(mb_.obs, value=True).value.view(-1))

        if multi_agent:
            step_n = [len(x) for x in rollout.reward]
            reward = np.concatenate(rollout.reward)
            ret, adv = calc_adv_multi_agent(np.concatenate(step_id_), reward, c.gamma, value_=value_, lam=c.lam)
            rollout.update(obs=step_obs, ret=split(ret, step_n))
            if c.use_critic:
                rollout.update(value=split(value_[:len(ret)], step_n), adv=split(adv, step_n))
        else:
            reward = rollout.reward
            ret, adv = calc_adv(reward, c.gamma, value_, c.lam)
            rollout.update(obs=step_obs, ret=ret)
            if c.use_critic:
                rollout.update(value=value_[:len(ret)], adv=adv)

        log = c.get_log_ii(ii)
        log(**stats)
        log(
            reward_mean=np.mean(reward),
            value_mean=np.mean(value_) if c.use_critic else None,
            ret_mean=np.mean(ret),
            adv_mean=np.mean(adv) if c.use_critic else None,
            explained_variance=explained_variance(value_[:len(ret)], ret) if c.use_critic else None
        )
        log(rollout_end_time=time() - t_start)
        return rollout

    def on_step_end(c, stats={}):
        c.log_stats(stats, print_time=True)
        c.log('')

    def on_train_end(c):
        if c._results is not None:
            c.save_train_results(c._results)

        save_path = c.save_state(c._i, c.get_state(c._model, c._opt, c._i))
        if c.tb:
            c._writer.close()
        if hasattr(c._env, 'close'):
            c._env.close()

    def train(c):
        c.on_train_start()
        while c._i < c.n_steps:
            c.on_step_start()
            with torch.no_grad():
                rollouts = c.rollouts()
            gd_stats = {}
            if len(rollouts.obs):
                t_start = time()
                c._alg.optimize(rollouts)
                gd_stats.update(gd_time=time() - t_start)
            c.on_step_end(gd_stats)
            c._i += 1
        c.on_step_start() # last step
        with torch.no_grad():
            rollouts = c.rollouts()
            c.on_step_end(gd_stats)
        c.on_train_end()

    def eval(c):
        c.setdefaults(alg='PPO')
        c._env = c.create_env()

        c._alg = (eval(c.alg) if isinstance(c.alg, str) else c.alg)(c)
        c.set_model()
        c._model.eval()
        c._results = pd.DataFrame(index=pd.Series(name='step'))
        c._writer_buffer = NamedArrays()

        kwargs = {'step' if isinstance(c.e, int) else 'path': c.e}
        step = c.set_state(c._model, opt=None, **kwargs)
        c.log('Loaded model from step %s' % step)

        c._run_start_time = time()
        c._i = 1
        for _ in range(c.n_steps):
            c.rollouts()
            if c.get('result_save'):
                c._results.to_csv(c.result_save)
            if c.get('vehicle_info_save'):
                np.savez_compressed(c.vehicle_info_save, **{k: v.values.astype(type(v.iloc[0])) for k, v in c._env.vehicle_info.iteritems()})
                if c.get('save_agent'):
                    np.savez_compressed(c.vehicle_info_save.replace('.npz', '_agent.npz'), **{k: v.values.astype(type(v.iloc[0])) for k, v in c._env.agent_info.iteritems()})
                c._env.sumo_paths['net'].cp(c.vehicle_info_save.replace('.npz', '.net.xml'))
            c._i += 1
            c.log('')
        if hasattr(c._env, 'close'):
            c._env.close()

    def run(c):
        c.log(format_yaml({k: v for k, v in c.items() if not k.startswith('_')}))
        c.setdefaults(n_rollouts_per_step=1)
        if c.e is not False:
            c.n_workers = 1
            c.setdefaults(use_ray=False, n_rollouts_per_worker=c.n_rollouts_per_step // c.n_workers)
            c.eval()
        else:
            c.setdefaults(device='cuda' if torch.cuda.is_available() else 'cpu')
            if c.get('use_ray', True) and c.n_rollouts_per_step > 1 and c.get('n_workers', np.inf) > 1:
                c.setdefaults(n_workers=c.n_rollouts_per_step, use_ray=True)
                c.n_rollouts_per_worker = c.n_rollouts_per_step // c.n_workers
                import ray
                try:
                    ray.init(num_cpus=c.n_workers, include_dashboard=False)
                except:
                    ray.init(num_cpus=c.n_workers, include_dashboard=False, _temp_dir=(Path.env('F') / 'tmp')._real)
                RemoteMain = ray.remote(type(c))
                worker_kwargs = c.get('worker_kwargs') or [{}] * c.n_workers
                assert c.n_workers % len(worker_kwargs) == 0
                c.log(f'Running {c.n_workers} with {len(worker_kwargs)} different worker kwargs')
                n_repeats = c.n_workers // len(worker_kwargs)
                worker_kwargs = [{**c, 'main': False, 'device': 'cpu', **args} for args in worker_kwargs for _ in range(n_repeats)]
                c._rollout_workers = [RemoteMain.remote(**kwargs, i_worker=i) for i, kwargs in enumerate(worker_kwargs)]
                ray.get([w.on_rollout_worker_start.remote() for w in c._rollout_workers])
            else:
                c.setdefaults(n_workers=1, n_rollouts_per_worker=c.n_rollouts_per_step, use_ray=False)
            assert c.n_workers * c.n_rollouts_per_worker == c.n_rollouts_per_step
            c.train()
