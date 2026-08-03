from 01_MNESIS_boilerplate import *

class HD_SNN(nn.Module):
    def __init__(self, opt, pattern_object):
        super().__init__()
        self.opt = opt
        self.target = pattern_object # the pattern object which create a function generating spikes
        self.target.init(opt)

        dropout = nn.Dropout(opt.dropout)

        lin = nn.Linear(opt.num_delay*opt.N_neuron, opt.N_neuron, bias=False)
        if self.opt.surrogate_name == 'FastSigmoid':
            spike_grad = surrogate.fast_sigmoid(slope=opt.alpha_surrogate)
        elif self.opt.surrogate_name == 'LeakySpikeOperator':
            spike_grad = surrogate.LSO(slope=opt.alpha_surrogate)
        elif self.opt.surrogate_name == 'ATan':
            spike_grad = surrogate.atan(alpha=opt.alpha_surrogate)
        elif self.opt.surrogate_name == 'SpikeRateEscape':
            spike_grad = surrogate.spike_rate_escape(slope=opt.alpha_surrogate)
        elif self.opt.surrogate_name == 'Sigmoid':
            spike_grad = surrogate.sigmoid(slope=opt.alpha_surrogate)

        lif = snn.Leaky(beta=torch.tensor(opt.lif_beta, dtype=torch.float32), 
                        threshold=torch.tensor(opt.lif_threshold, dtype=torch.float32),
                        learn_beta=opt.learn_beta, learn_threshold=opt.learn_threshold, output=False, # init_hidden=True, 
                        reset_mechanism=opt.reset_mechanism, spike_grad=spike_grad)

        net = nn.Sequential(OrderedDict([('lin', lin), ('dropout', dropout), ('lif', lif) ]))
        net = net.to(opt.device)
        self.net = net
          
    def forward_pass(self, input_spikes, reset_spikes=None):
        input_spikes = input_spikes.to(self.opt.device).detach()
        assert not input_spikes.requires_grad, "input_spikes still requires grad!"
        if reset_spikes is None: reset_spikes = torch.zeros_like(input_spikes)

        with torch.no_grad():
            snn_utils.reset(self.net)

        device, dtype = self.opt.device, torch.float32

        N_pattern = input_spikes.shape[0]
        N_time = input_spikes.shape[-1] # self.opt.N_time+2*self.opt.N_pretime
        current = torch.zeros(N_pattern, self.opt.N_neuron, N_time, device=device, dtype=dtype)
        spikes  = torch.zeros(N_pattern, self.opt.N_neuron, N_time, device=device, dtype=dtype)
        mem_rec = torch.zeros(N_pattern, self.opt.N_neuron, N_time, device=device, dtype=dtype)
        # Initalize membrane potential
        mem = self.net.lif.init_leaky() # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]

        for t in range(self.opt.num_delay, N_time):
            spike_window_A = spikes[:, :, (t - self.opt.num_delay):t] # recurrent spikes
            spike_window_B = input_spikes[:, :, (t - self.opt.num_delay):t] # excitatory input spikes
            spike_window_C = reset_spikes[:, :, (t - self.opt.num_delay):t] # inhibitory reset spikes
            spike_window = (spike_window_A + spike_window_B - spike_window_C).clamp(0, 1)
            # window   [N_pattern, N_neuron, num_delay]  ->  [N_pattern, N_neuron * num_delay]
            raveled_spks = spike_window.reshape(N_pattern, self.opt.N_neuron * self.opt.num_delay)
            cur = self.net.lin(raveled_spks)                     # pyright: ignore[reportCallIssue] # shape [1, N_neuron]
            cur = self.net.dropout(cur) # pyright: ignore[reportCallIssue]

            spk, mem = self.net.lif(cur, mem)          # pyright: ignore[reportCallIssue] # spk, mem shape [1, N_neuron]
            
            current[:, :, t] = cur
            spikes[:, :, t] = spk
            mem_rec[:, :, t] = mem

        return current, mem_rec, spikes

    def get_W_init(self):
        \"\"\"
        Initialise the linear layer weights via one of two methods, controlled
        by the boolean flags ``opt.do_pinv`` and ``opt.do_deconv``.

        TARGET SIGNAL
        -------------
        If ``opt.do_deconv`` is True, the regression target is the
        *deconvolved* spike train (Eq. deconv_target):

            I*(t) = s*(t) - beta * s*(t-1)

        This is the input current that would produce a membrane Dirac at each
        target spike time, exactly compensating for the IIR lowpass of the LIF
        membrane (H(z) = 1 / (1 - beta * z^{-1})).  It significantly reduces
        spurious early spikes at high beta.  When ``do_deconv`` is False, the
        raw binary target s*(t) is used instead (valid approximation for small beta).

        METHOD A — pseudo-inverse  (``do_pinv=True``)
        -----------------------------------------------
        Exact minimum-ell_2-norm solution  W* = I* @ pinv(C).
        Requires an SVD of the context matrix C of shape (batch, N*D), which
        has complexity O(batch * (N*D)^2).  This is the bottleneck for large
        networks or long patterns:

            batch = N_pattern * (N_time - num_delay)
            N*D   = N_neuron * num_delay

        For the default parameters (N=256, D=41, T=512, M=8):
            batch = 8 * 471 = 3768,  N*D = 10496
        The SVD dominates; use ``do_pinv=False`` to skip it.

        METHOD B — normalised cross-correlation  (``do_pinv=False``)
        --------------------------------------------------------------
        Closed-form Hebbian approximation (Eq. init in the paper):

            W_{j,i,d} = (1 / norm) * sum_{mu,t} I*_j(t) * s_i(t-d)

        where norm = mean diagonal of C^T C ≈ N*D*p_A*batch.
        This is exact when consecutive context windows are orthogonal, which
        holds when p_A << 1 (off-diagonal / diagonal of the Gram matrix ≈ p_A).
        At p_A = 10^{-3} (paper value) the approximation error is < 0.1 %.
        Cost: two matrix multiplies, no SVD — typically 10-100x faster.

        RECOMMENDATION
        --------------
        - Use do_pinv=True  for small N, D, T  or when exact init matters.
        - Use do_pinv=False for large networks; the approximation is excellent
          at the sparse firing rates used in this paper.
        - Always use do_deconv=True when beta >= 0.5.
        \"\"\"
        # ------------------------------------------------------------------
        # 1. Build context windows  C  of shape (batch, N_neuron * num_delay)
        #    Each row is the flattened spike history [s(t-D), ..., s(t-1)]
        #    for one (pattern, time) pair.
        # ------------------------------------------------------------------
        target = self.target() # take one instance of the target pattern, shape (N_pattern, N_neuron, N_time)
        windows = target[:, :, :-1].unfold(
            dimension=2, size=self.opt.num_delay, step=1
        )                                                   # (N_pattern, N_neuron, T-D, D)
        windows  = windows.permute(0, 2, 1, 3).contiguous()
        batch    = self.opt.N_pattern * (self.opt.N_time - self.opt.num_delay)
        contexts = windows.reshape(
            batch, self.opt.N_neuron * self.opt.num_delay
        )                                                   # (batch, N_neuron*D)
            
        # ------------------------------------------------------------------
        # 2. Build regression targets  I*  of shape (batch, N_neuron)
        # ------------------------------------------------------------------
        if self.opt.do_deconv:
            # I*(t) = s*(t) - beta * s*(t-1)
            # torch.roll(..., 1, dims=-1) shifts the time axis right by 1,
            # placing s*(t-1) at position t.  The value at t=0 wraps around
            # from t=T-1, but it is discarded when we slice [:, :, num_delay:].
            deconvolved = target - self.opt.lif_beta * torch.roll(target, 1, dims=-1)
            raw_targets = deconvolved[:, :, self.opt.num_delay:]
        else:
            raw_targets = target[:, :, self.opt.num_delay:]

        targets = raw_targets.permute(0, 2, 1).reshape(
            batch, self.opt.N_neuron
        )                                                   # (batch, N_neuron)

        # ------------------------------------------------------------------
        # 3. Solve for W
        # ------------------------------------------------------------------
        if self.opt.do_pinv:
            try:

                contexts_cpu = contexts.cpu()
                targets_cpu = targets.cpu()
                                
                # --- Method A: pseudo-inverse ---
                # W* = pinv(C) @ I*   =>  shape (N_neuron*D, N_neuron)
                X_pinv = torch.linalg.pinv(contexts_cpu)            # (N_neuron*D, batch)
                W_init = torch.matmul(X_pinv, targets_cpu)          # (N_neuron*D, N_neuron)
                W_init = W_init.transpose(0, 1)                 # (N_neuron, N_neuron*D)
                W_init = W_init.to(contexts.device)

            except RuntimeError as e:
                # Add debugging
                print(\"Error during pseudo-inverse computation:\", e)
                print(f\"Contexts shape: {contexts.shape}\")
                print(f\"Targets shape: {targets.shape}\")
                print(f\"Contexts NaN count: {torch.isnan(contexts).sum().item()}\")
                print(f\"Contexts inf count: {torch.isinf(contexts).sum().item()}\")
                print(f\"Contexts min/max: {contexts.min().item():.6f} / {contexts.max().item():.6f}\")
                # Print some sample values
                print(\"Sample contexts values:\", contexts.flatten()[:10])
                # Check if targets also have issues
                print(f\"Targets NaN count: {torch.isnan(targets).sum().item()}\")
                raise e
        else:
            # --- Method B: normalised cross-correlation ---
            # W ≈ (I*^T @ C) / norm
            # norm = mean diagonal of (C^T C)
            #      = mean over columns of sum_rows(c^2)
            #      ≈ N_neuron * num_delay * p_A * batch  (sparse Gram approx)
            # Using the exact empirical norm is more robust than the
            # theoretical approximation, especially at higher firing rates.
            norm = (contexts * contexts).sum(dim=0).mean().clamp(min=1e-8)
            W_init = torch.matmul(targets.t(), contexts) / norm
                                                            # (N_neuron, N_neuron*D)

        return W_init
        
    def update_weight(self):
        with torch.no_grad():            
            W_init = self.get_W_init()
            self.net.lin.weight.copy_(W_init)  # pyright: ignore
            
    def get_input_spikes(self, p_A=None, N_pretime=None, N_trigger_time=None, N_time=None):
        if p_A is None: p_A = self.opt.p_A 
        if N_pretime is None: N_pretime = self.opt.N_pretime
        if N_trigger_time is None: N_trigger_time = self.opt.num_delay
        if N_time is None: N_time = self.opt.N_time

        # clamp spikes to start the chain
        # 1/ empty spike list
        input_spikes = torch.zeros((self.opt.N_pattern, self.opt.N_neuron, N_time+2*N_pretime))
        # 2/ spontaneous (random) activity
        input_spikes[:, :, :N_pretime] = torch.bernoulli(p_A * torch.ones((self.opt.N_pattern, self.opt.N_neuron, N_pretime)))
        # 3/ target activity           
        input_spikes[:, :, N_pretime:(N_pretime+N_trigger_time)] = self.target()[:, :, :N_trigger_time]

        return input_spikes.to(self.opt.device).detach()

    def learn_model(self, verbose=True):
        
        if self.opt.loss_name == 'SpikeF1scoreLoss':
            loss_fn = SpikeF1scoreLoss()
        elif self.opt.loss_name == 'MSELoss':
            loss_fn = nn.MSELoss()

        self.net = self.net.to(device)
        
        
        optimizer_dict = dict(lr=self.opt.base_lr) #, weight_decay=self.opt.weight_decay)
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

        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    self.opt.num_warmup_epochs, self.opt.num_epochs, self.opt.final_lr/self.opt.base_lr)

        loss_val, precision, recall, f1_score = [], [], [], []
        log_interval = max(self.opt.num_epochs // 64, 1)

        for i_step in range(self.opt.num_epochs):
            self.net.train()
            target = self.target()
            # start optimization
            optimizer.zero_grad()
            input_spikes = self.get_input_spikes().detach()
            _, _, spikes = self.forward_pass(input_spikes)
            loss_train = loss_fn(spikes[:, :, (self.opt.N_pretime+self.opt.num_delay):(-self.opt.N_pretime)], 
                                 target[:, :, self.opt.num_delay:])
            loss_train.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                self.net.eval()
                # draw causes (SMs) - hidden to the observer
                input_spikes = self.get_input_spikes()
                _, _, spikes = self.forward_pass(input_spikes)
                spikes_ = spikes[:, :, (self.opt.N_pretime+self.opt.num_delay):(-self.opt.N_pretime)]
                target_ = target[:, :, self.opt.num_delay:]
                loss_val_ = loss_fn(spikes_, target_)
                loss_val.append(loss_val_.item())
                precision_, recall_, f1_score_ = get_scores(spikes_, target_)
                precision.append(precision_.cpu()) 
                recall.append(recall_.cpu())
                f1_score.append(f1_score_.cpu())

            if verbose and ((i_step + 1) % log_interval == 0):
                print(f'Train Epoch [{i_step+1:06d}/{self.opt.num_epochs:06d}]\\t| Loss = {np.mean(loss_val):.3e}\\t| precision = {np.mean(precision):.3f}\\t| recall = {np.mean(recall):.3f}\\t| f1_score = {np.mean(f1_score):.3f}\\t| ')
                loss_val, precision, recall, f1_score = [], [], [], []

def load(opt, model_filename, pattern_object=StochasticSpikingPattern()):
    hd = HD_SNN(opt, pattern_object=pattern_object)
    hd.net.to(hd.opt.device)
    model_state_dict = torch.load(model_filename, map_location=torch.device(hd.opt.device))
    hd.net.load_state_dict(model_state_dict)
    hd.net.eval()
    return hd
