"""Core script for the MNSESI library.

Defines Spiking-pattern generators and the HD_SNN network class for MNESIS experiments with training and inference methods.

"""

from mnesis_boilerplate import (
    torch, np, nn, OrderedDict, surrogate, snn, snn_utils,
    get_scores, get_cosine_schedule_with_warmup, SpikeF1scoreLoss,
    DEBUG, i_pattern, phi, figpath, printfig, flip_bits, Params,
)


class SpikingPattern:
    """Generate a frozen spiking pattern based on a Bernoulli process."""

    def __init__(self):
        self.desc = "Frozen Spike pattern generator"
        self.is_periodic = False

    def init(self, opt, verbose=False):
        self.opt = opt
        frozen_target_generator = torch.Generator()  # used once to generate the target pattern
        frozen_target_generator.manual_seed(opt.seed)
        p_bias = opt.p_A * torch.ones((opt.N_pattern, opt.N_neuron, opt.N_time))
        self.frozen_target = torch.bernoulli(p_bias, generator=frozen_target_generator)
        self.frozen_target = self.frozen_target.float()
        self.frozen_target = self.frozen_target.to(opt.device)
        if verbose:
            print(f"Target pattern generated with shape {self.frozen_target.shape} and mean {self.frozen_target.mean().item():.3e}")

    def __call__(self):
        return self.frozen_target


class StochasticSpikingPattern(SpikingPattern):
    """A stochastic spiking pattern generator.

    A stochastic pattern generator that creates variable realizations
    of patterns while preserving average firing rates.

    This class extends SpikingPattern by adding stochastic variability through
    a "balanced bit flipping" operation. Each call to __call__() returns a new realization
    of the base pattern with bits flipped independently with probability p_flip.
    The marginal frequency is exactly preserved while the pattern structure
    is stochastically modified.
    """

    def __init__(self): 
        """
        A stochastic spiking pattern generator that creates variable realizations
        of patterns while preserving average firing rates.

        This class extends SpikingPattern by adding stochastic variability through
        a "balanced bit flipping" operation. 
        """
        super().__init__()
        self.desc = "Stochastic spike pattern generator"

    def __call__(self, seed=None, verbose=False):
        """
        Generate a stochastic realization of the spiking pattern.

        Returns a new version of the base pattern where each bit has been
        independently flipped with probability self.p_flip. Each call to __call__() returns a new realization
        of the base pattern with bits flipped independently with probability p_flip. The flip operation
        preserves the marginal frequency while introducing temporal and spatial
        variability in the pattern structure.

        Returns:
            torch.Tensor: Stochastic realization of the spiking pattern
                         with same dimensions as base pattern
        """
        return flip_bits(self.frozen_target, p_flip=self.opt.p_flip, seed=seed, verbose=verbose)


class HD_SNN(nn.Module):
    def __init__(self, opt, pattern_object):
        super().__init__()
        self.opt = opt
        self.target = pattern_object
        self.target.init(opt)

        dropout = nn.Dropout(opt.dropout)
        lin = nn.Linear(opt.num_delay*opt.N_neuron, opt.N_neuron, bias=False)
        
        if self.opt.surrogate_name == 'FastSigmoid':
            spike_grad = surrogate.fast_sigmoid(slope=opt.alpha_surrogate)
        elif self.opt.surrogate_name == 'LeakySpikeOperator':
            spike_grad = surrogate.LSO(slope=opt.alpha_surrogate)
        elif self.opt.surrogate_name == 'ATan':
            spike_grad = surrogate.atan(alpha=opt.alpha_surrogate)
        elif self.opt.surrogate_name == 'SpikeRateEscape': # Placeholder from original source typo likely
            spike_grad = surrogate.spike_rate_escape(slope=opt.alpha_surrogate)
        elif self.opt.surrogate_name == 'Sigmoid':
            spike_grad = surrogate.sigmoid(slope=opt.alpha_surrogate)
        else:
            spike_grad = surrogate.fast_sigmoid(slope=opt.alpha_surrogate)

        lif = snn.Leaky(beta=torch.tensor(opt.lif_beta, dtype=torch.float32), 
                        threshold=torch.tensor(opt.lif_threshold, dtype=torch.float32),
                        learn_beta=opt.learn_beta, learn_threshold=opt.learn_threshold, output=False,
                        reset_mechanism=opt.reset_mechanism, spike_grad=spike_grad)

        self.net = nn.Sequential(OrderedDict([('lin', lin), ('dropout', dropout), ('lif', lif)]))
        self.net = self.net.to(opt.device)
          
    def forward_pass(self, input_spikes, reset_spikes=None):
        input_spikes = input_spikes.to(self.opt.device).detach()
        if reset_spikes is None: reset_spikes = torch.zeros_like(input_spikes)

        with torch.no_grad():
            snn_utils.reset(self.net)

        device, dtype = self.opt.device, torch.float32
        N_pattern = input_spikes.shape[0]
        N_time = input_spikes.shape[-1]
        current = torch.zeros(N_pattern, self.opt.N_neuron, N_time, device=device, dtype=dtype)
        spikes  = torch.zeros(N_pattern, self.opt.N_neuron, N_time, device=device, dtype=dtype)
        mem_rec = torch.zeros(N_pattern, self.opt.N_neuron, N_time, device=device, dtype=dtype)
        mem = self.net.lif.init_leaky()

        for t in range(self.opt.num_delay, N_time):
            spike_window_A = spikes[:, :, (t - self.opt.num_delay):t]
            spike_window_B = input_spikes[:, :, (t - self.opt.num_delay):t]
            spike_window_C = reset_spikes[:, :, (t - self.opt.num_delay):t]
            spike_window = (spike_window_A + spike_window_B - spike_window_C).clamp(0, 1)
            raveled_spks = spike_window.reshape(N_pattern, self.opt.N_neuron * self.opt.num_delay)
            cur = self.net.lin(raveled_spks)
            cur = self.net.dropout(cur)
            spk, mem = self.net.lif(cur, mem)
            current[:, :, t] = cur
            spikes[:, :, t] = spk
            mem_rec[:, :, t] = mem

        return current, mem_rec, spikes

    def get_W_init(self):
        target = self.target()
        windows = target[:, :, :-1].unfold(dimension=2, size=self.opt.num_delay, step=1)
        windows  = windows.permute(0, 2, 1, 3).contiguous()
        batch    = self.opt.N_pattern * (self.opt.N_time - self.opt.num_delay)
        contexts = windows.reshape(batch, self.opt.N_neuron * self.opt.num_delay)
            
        if self.opt.do_deconv:
            deconvolved = target - self.opt.lif_beta * torch.roll(target, 1, dims=-1)
            raw_targets = deconvolved[:, :, self.opt.num_delay:]
        else:
            raw_targets = target[:, :, self.opt.num_delay:]

        targets = raw_targets.permute(0, 2, 1).reshape(batch, self.opt.N_neuron)

        if self.opt.do_pinv:
            contexts_cpu, targets_cpu = contexts.cpu(), targets.cpu()
            X_pinv = torch.linalg.pinv(contexts_cpu)
            W_init = torch.matmul(X_pinv, targets_cpu)
            W_init = W_init.transpose(0, 1).to(contexts.device)
        else:
            norm = (contexts * contexts).sum(dim=0).mean().clamp(min=1e-8)
            W_init = torch.matmul(targets.t(), contexts) / norm

        return W_init
        
    def update_weight(self):
        with torch.no_grad():            
            W_init = self.get_W_init()
            self.net.lin.weight.copy_(W_init)
            
    def get_input_spikes(self, target=None, p_A=None, N_pretime=None, N_trigger_time=None, N_time=None):
        """
        generate the trigger input spikes for the network, including pre-time spontaneous activity and the target pattern.
        
        """
        if p_A is None: p_A = self.opt.p_A 
        if N_pretime is None: N_pretime = self.opt.N_pretime
        if N_trigger_time is None: N_trigger_time = self.opt.num_delay
        if N_time is None: N_time = self.opt.N_time

        input_spikes = torch.zeros((self.opt.N_pattern, self.opt.N_neuron, N_time+2*N_pretime))
        input_spikes[:, :, :N_pretime] = torch.bernoulli(p_A * torch.ones((self.opt.N_pattern, self.opt.N_neuron, N_pretime)))
        if target is None:
            target = self.target()
        input_spikes[:, :, N_pretime:(N_pretime+N_trigger_time)] = target[:, :, :N_trigger_time]
        return input_spikes.to(self.opt.device).detach()

    def learn_model(self, verbose=True):
        if self.opt.loss_name == 'SpikeF1scoreLoss':
            loss_fn = SpikeF1scoreLoss()
        elif self.opt.loss_name == 'MSELoss':
            loss_fn = nn.MSELoss()

        self.net = self.net.to(self.opt.device)
        optimizer_dict = dict(lr=self.opt.base_lr)
        if self.opt.optimizer=='adam': 
            optimizer = torch.optim.Adam(self.net.parameters(), betas=(1-self.opt.delta1, 1-self.opt.delta2), **optimizer_dict)
        elif self.opt.optimizer=='adamw': 
            optimizer = torch.optim.AdamW(self.net.parameters(), betas=(1-self.opt.delta1, 1-self.opt.delta2), **optimizer_dict)
        elif self.opt.optimizer=='sparseadam': 
            optimizer = torch.optim.AdamW(self.net.parameters(), betas=(1-self.opt.delta1, 1-self.opt.delta2), **optimizer_dict)
        elif self.opt.optimizer=='sgd': 
            optimizer = torch.optim.SGD(self.net.parameters(),  momentum=1-self.opt.delta1, dampening=1-self.opt.delta2, **optimizer_dict)
        elif self.opt.optimizer=='rmsprop': 
            optimizer = torch.optim.RMSprop(self.net.parameters(), momentum=1-self.opt.delta1, alpha=1-self.opt.delta2, **optimizer_dict)
        elif self.opt.optimizer=='adadelta': 
            optimizer = torch.optim.Adadelta(self.net.parameters(), rho=1-self.opt.delta1, **optimizer_dict)
        else:
            raise(ValueError(f'Unknown optimizer {self.opt.optimizer}'))

        scheduler = get_cosine_schedule_with_warmup(optimizer, self.opt.num_warmup_epochs, self.opt.num_epochs, self.opt.final_lr/self.opt.base_lr)

        loss_val, precision, recall, f1_score = [], [], [], []
        log_interval = max(self.opt.num_epochs // 64, 1)

        for i_step in range(self.opt.num_epochs):
            self.net.train()
            # the pattern that we wish to memorize
            target = self.target() # NOTE: we assume that the pattern generator can generate a new pattern each time it is called
            # the input spikes that we feed to the network, which includes pre-time spontaneous activity (padding) and the target pattern just for the trigger window
            input_spikes = self.get_input_spikes(target=target).detach()
            optimizer.zero_grad()
            # the optimal output spikes that the network produces in response to the input spikes
            _, _, output_spikes = self.forward_pass(input_spikes)
            loss_train = loss_fn(output_spikes[:, :, (self.opt.N_pretime+self.opt.num_delay):(self.opt.N_time-self.opt.N_pretime)], 
                                 target[:, :, self.opt.num_delay:])
            loss_train.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                self.net.eval()
                # the pattern that we wish to memorize
                target = self.target()
                # the input spikes 
                input_spikes = self.get_input_spikes(target=target).detach()
                # the optimal output spikes that the network produces in response to the input spikes
                _, _, output_spikes = self.forward_pass(input_spikes)
                output_spikes_trimmed = output_spikes[:, :, (self.opt.N_pretime+self.opt.num_delay):(self.opt.N_time-self.opt.N_pretime)]
                input_target_trimmed = target[:, :, self.opt.num_delay:]
                loss_val_ = loss_fn(output_spikes_trimmed, input_target_trimmed)
                loss_val.append(loss_val_.item())
                precision_, recall_, f1_score_ = get_scores(output_spikes_trimmed, input_target_trimmed)
                precision.append(precision_.cpu()) 
                recall.append(recall_.cpu())
                f1_score.append(f1_score_.cpu())

            if verbose and ((i_step + 1) % log_interval == 0):
                print(f'Train Epoch [{i_step+1:06d}/{self.opt.num_epochs:06d}]\t| Loss = {np.mean(loss_val):.3e}\t| precision = {np.mean(precision):.3f}\t| recall = {np.mean(recall):.3f}\t| f1_score = {np.mean(f1_score):.3f}\t| ')
                loss_val, precision, recall, f1_score = [], [], [], []

def load(opt, model_filename, pattern_object=None):
    if pattern_object is None:
        pattern_object = StochasticSpikingPattern()

    hd = HD_SNN(opt, pattern_object=pattern_object)
    hd.net.to(hd.opt.device)
    model_state_dict = torch.load(model_filename, map_location=torch.device(hd.opt.device))
    hd.net.load_state_dict(model_state_dict)
    hd.net.eval()
    return hd
