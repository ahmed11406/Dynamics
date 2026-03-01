function out = GABI_FBS( ...
    N, c, p, A0_1, Ap_j, C0_1, Y, Mb, V0_1, q0, qdot, ...
    Wprop, tau_bar, Wapp, rmax, g)
% Optimized GABI_FBS:
% - iterative pre/post order traversals (no recursion)
% - no nchoosek in hot loops (precomputed binomials)
% - fast SE(3) exp (closed form) instead of expm
% - cached/incremental M-derivatives for all bodies/orders
% - compute Vbias once per order
% - LDL of MA(base) once, reused every order

% -----------------------
% knobs
% -----------------------
DEBUG        = false;   % set true to enable asserts (slower)
RETURN_CELLS = true;    % set false for fastest + least memory (returns numeric arrays)

if DEBUG
    assert(iscell(c) && numel(c)==N);
    assert(iscell(Ap_j) && numel(Ap_j)==N);
    assert(iscell(Y) && numel(Y)==N);
    assert(iscell(Mb) && numel(Mb)==N);
    assert(iscell(Wapp) && numel(Wapp)==N);
    assert(all(size(Wprop)==[6, rmax+1]));
    assert(all(size(tau_bar)==[N, rmax+1]));
    assert(all(size(C0_1)==[4 4]) && all(size(A0_1)==[4 4]));
    assert(numel(q0)==N && numel(qdot)==N);
    assert(all(size(V0_1)==[6 1]));
end

G0 = [0;0;0; 0;0;-g];

% -----------------------
% Precompute binomial table B(n+1,k+1)=C(n,k), up to n=rmax+1
% -----------------------
nMax = rmax + 1;
B = zeros(nMax+2, nMax+2);
B(1,1) = 1;
for n = 1:(nMax+1)
    B(n+1,1)   = 1;
    B(n+1,n+1) = 1;
    for k = 1:(n-1)
        B(n+1,k+1) = B(n,k+1) + B(n,k);
    end
end

% -----------------------
% Build preorder and postorder traversals (iterative)
% -----------------------
preOrder = zeros(N,1);
stk = zeros(N,1); sp = 1; stk(sp) = 1; m = 0;
while sp > 0
    j = stk(sp); sp = sp - 1;
    m = m + 1; preOrder(m) = j;
    kids = c{j};
    % push kids (reverse keeps original order if you care)
    for ii = numel(kids):-1:1
        sp = sp + 1; stk(sp) = kids(ii);
    end
end
preOrder = preOrder(1:m);

postOrder = zeros(N,1);
stk1 = zeros(N,1); stk2 = zeros(N,1);
sp1 = 1; stk1(sp1) = 1; sp2 = 0;
while sp1 > 0
    j = stk1(sp1); sp1 = sp1 - 1;
    sp2 = sp2 + 1; stk2(sp2) = j;
    kids = c{j};
    for ii = 1:numel(kids)
        sp1 = sp1 + 1; stk1(sp1) = kids(ii);
    end
end
% reverse stk2
for i = 1:sp2
    postOrder(i) = stk2(sp2 - i + 1);
end

% -----------------------
% Allocate contiguous storage
% -----------------------
A0  = zeros(4,4,N);
F   = zeros(4,4,N);
C0  = zeros(4,4,N);

Sder = zeros(6, rmax+2, N);     % S^(0..rmax+1)
Vder = zeros(6, rmax+2, N);     % V^(0..rmax+1)

M0   = zeros(6,6,N);
MA   = zeros(6,6,N);
Dsc  = nan(N,1);

WA   = zeros(6, rmax+1, N);     % WA^(0..rmax)

qder = zeros(N, rmax+3);        % q^(0..rmax+2)
qder(:,1) = q0(:);
qder(:,2) = qdot(:);
qder(1,:) = NaN;

qhigh = nan(N, rmax+1);         % q^(r+2)
qtil  = nan(N, rmax+1);         % tilde q^(r+2)

% -----------------------
% Forward poses + S^(0) + V^(0)
% -----------------------
A0(:,:,1) = A0_1;
F(:,:,1)  = C0_1;
C0(:,:,1) = F(:,:,1) * A0(:,:,1);

Sder(:,1,1) = 0;
Vder(:,1,1) = V0_1;

for idx = 2:numel(preOrder)
    j   = preOrder(idx);
    pj  = p(j);

    A0(:,:,j) = A0(:,:,pj) * Ap_j{j};

    % F_j = F_p * exp([Y_j] q_j)
    F(:,:,j)  = F(:,:,pj) * expSE3_fast(Y{j}, q0(j));

    C0(:,:,j) = F(:,:,j) * A0(:,:,j);

    % S^(0) = Ad(F_j) * Y_j (direct multiply, no 6x6 Ad build)
    Sder(:,1,j) = Ad_mul_Twist(F(:,:,j), Y{j});

    % V^(0)
    Vder(:,1,j) = Vder(:,1,pj) + Sder(:,1,j) * qdot(j);
end

% -----------------------
% M0 in inertial frame (once)
% -----------------------
for j = 1:N
    M0(:,:,j) = transformInertiaToWorld(C0(:,:,j), Mb{j});
end

% -----------------------
% Articulated inertia MA (postorder), uses S^(0)
% -----------------------
for idx = 1:numel(postOrder)
    j = postOrder(idx);
    kids = c{j};

    MAj = M0(:,:,j);
    for ii = 1:numel(kids)
        i  = kids(ii);
        Si = Sder(:,1,i);
        Mi = MA(:,:,i);

        Ui = Mi * Si;
        Di = Si.' * Ui;
        Dsc(i) = Di;

        UiDi = Ui / Di;
        MAj  = MAj + (Mi - (UiDi * Ui.'));
    end
    MA(:,:,j) = MAj;
end

% Factorize base MA once (SPD expected)
[L0, D0, perm0] = ldl(MA(:,:,1), 'vector');

% -----------------------
% Caches for incremental M-derivatives
% Mder(:,:,:,rr+1) = M^(rr)
% rr = 0..rmax+1 needed (because Pi uses M^(r+1))
% -----------------------
Mder = zeros(6,6,N, rmax+3);
Mder(:,:,:,1) = M0;
Mmax = 0; % highest rr computed

% Workspace per order (small)
Vbias   = zeros(6,N);
Pi_bias = zeros(6,N);

Smax = 0; % highest S order computed (k), Sder contains S^(0) already

% =========================================================
% MAIN progressive loop
% =========================================================
for r = 0:rmax
    % ---- ensure S up to order (r+1)
    kNeed = r + 1;
    for k = (Smax+1):kNeed
        % S^(k) for all non-base nodes in preorder
        for idx = 2:numel(preOrder)
            j = preOrder(idx);

            Sk = zeros(6,1);
            % S^(k) = sum_{l=0}^{k-1} C(k-1,l) ad_{V^(l)} S^(k-1-l)
            % coeff = C(k-1,l) = B(k, l+1)
            for l = 0:(k-1)
                coeff = B(k, l+1);
                V_l   = Vder(:, l+1, j);      % V^(l)
                S_prev= Sder(:, k-l, j);      % S^(k-1-l) at col (k-1-l)+1 = k-l
                Sk    = Sk + coeff * ad_mul_Twist(V_l, S_prev);
            end
            Sder(:, k+1, j) = Sk;
        end
    end
    Smax = max(Smax, kNeed);

    % ---- ensure M up to order (r+1) when needed (r>0 uses Mr and Pi_bias)
    if r > 0
        rrNeed = r + 1; % need M^(r+1) for Pi_bias with m=r+1
        for rr = (Mmax+1):rrNeed
            % M^(rr) = - sum_{k=0}^{rr-1} C(rr-1,k) ( M^(rr-1-k) ad(V^(k)) + ad(V^(k))' M^(rr-1-k))
            for j = 1:N
                Mrr = zeros(6,6);
                for k = 0:(rr-1)
                    coeff = B(rr, k+1); % C(rr-1,k)
                    Mprev = Mder(:,:,j, rr-k); % (rr-1-k)+1 = rr-k
                    Vk    = Vder(:, k+1, j);
                    adVk  = ad_mat(Vk);
                    Mrr   = Mrr - coeff*(Mprev*adVk + adVk.'*Mprev);
                end
                Mder(:,:,j, rr+1) = Mrr;
            end
        end
        Mmax = max(Mmax, rrNeed);
    end

    % ---- compute Vbias for all nodes once (used in backward+forward)
    % Vbias = sum_{a=1}^{r+1} C(r+1,a) S^(a) q^(r-a+2)
    Vbias(:,1) = 0;
    for idx = 2:numel(preOrder)
        j = preOrder(idx);
        vb = zeros(6,1);
        for a = 1:(r+1)
            coeff = B(r+2, a+1);          % C(r+1,a)
            qcol  = (r - a + 3);          % q^(r-a+2) is in qder(:, (r-a+2)+1)
            vb    = vb + coeff*( Sder(:, a+1, j) * qder(j, qcol) );
        end
        Vbias(:,j) = vb;
    end

    % ---- compute Pi_bias for all bodies once when r>0
    if r > 0
        m = r + 1;
        for j = 1:N
            pib = zeros(6,1);
            for k = 0:r
                coeff = B(m+1, k+1);                   % C(m,k)
                Mk    = Mder(:,:,j, (m-k)+1);          % M^(m-k)
                pib   = pib + coeff*(Mk * Vder(:,k+1,j));
            end
            Pi_bias(:,j) = pib;
        end
    end

    % =====================================================
    % Backward: WA^(r) and qtil^(r+2) (postorder)
    % =====================================================
    for idx = 1:numel(postOrder)
        j = postOrder(idx);
        kids = c{j};
        V0j = Vder(:,1,j);

        if r == 0
            % WA^(0) isolated
            MV   = M0(:,:,j) * V0j;
            WAj  = minus_adT_mul(V0j, MV) ...
                 - Wapp{j}(:,1) ...
                 - (M0(:,:,j) * G0);

            % add children
            for ii = 1:numel(kids)
                i   = kids(ii);
                Si  = Sder(:,1,i);
                Sdi = Sder(:,2,i);
                MiA = MA(:,:,i);
                WiA = WA(:,1,i);
                Di  = Dsc(i);
                qd  = qder(i,2);

                qtil(i,1) = (1/Di) * ( tau_bar(i,1) - Si.'*( MiA*(Sdi*qd) + WiA ) );
                WAj       = WAj + ( WiA + MiA*(Si*qtil(i,1) + Sdi*qd) );
            end
            WA(:,1,j) = WAj;

        else
            Mr    = Mder(:,:,j, r+1); % M^(r)
            WAj_r = -Wapp{j}(:,r+1) - (Mr*G0) + Pi_bias(:,j);

            for ii = 1:numel(kids)
                i   = kids(ii);
                MiA = MA(:,:,i);
                Si  = Sder(:,1,i);
                Di  = Dsc(i);

                Vb  = Vbias(:,i);

                % tau_tilde^(r)
                tau_tilde = 0;
                for k = 0:(r-1)
                    coeff = B(r+1, k+1);             % C(r,k)
                    Sd    = Sder(:, (r-k)+1, i);     % S^(r-k)
                    Vk1   = Vder(:, k+2, i);         % V^(k+1)
                    WAk   = WA(:, k+1, i);           % WA^(k)
                    tau_tilde = tau_tilde + coeff * ( Sd.' * (MiA*Vk1 + WAk) );
                end

                qtil(i,r+1) = (1/Di) * ( ...
                    tau_bar(i,r+1) - tau_tilde ...
                    - Si.'*( MiA*Vb + WA(:,r+1,i) ) );

                WAj_r = WAj_r + ( WA(:,r+1,i) + MiA*(Si*qtil(i,r+1) + Vb) );
            end

            WA(:,r+1,j) = WAj_r;
        end
    end

    % =====================================================
    % Forward: V^(r+1), q^(r+2), propagate V^(r+1)
    % =====================================================
    rhs = Wprop(:,r+1) - WA(:,r+1,1);
    Vder(:, r+2, 1) = solve_ldl_fact(L0, D0, perm0, rhs);

    for idx = 2:numel(preOrder)
        j  = preOrder(idx);
        pj = p(j);

        Vp = Vder(:, r+2, pj);
        Sj = Sder(:,1,j);
        Di = Dsc(j);

        qhigh(j,r+1) = -(1/Di) * ( Sj.' * (MA(:,:,j) * Vp) ) + qtil(j,r+1);
        Vder(:, r+2, j) = Vp + Vbias(:,j) + Sj*qhigh(j,r+1);
    end

    % store q^(r+2)
    qder(:, r+3) = qhigh(:, r+1);
end

% -----------------------
% Pack outputs
% -----------------------
if RETURN_CELLS
    out.Vder = squeeze(num2cell(Vder, [1 2])).';
    out.Sder = squeeze(num2cell(Sder, [1 2])).';
    out.WA   = squeeze(num2cell(WA,   [1 2])).';
    out.MA   = squeeze(num2cell(MA,   [1 2])).';
    out.C0   = squeeze(num2cell(C0,   [1 2])).';
    out.F    = squeeze(num2cell(F,    [1 2])).';
    out.A0   = squeeze(num2cell(A0,   [1 2])).';
else
    out.Vder = Vder;
    out.Sder = Sder;
    out.WA   = WA;
    out.MA   = MA;
    out.C0   = C0;
    out.F    = F;
    out.A0   = A0;
end

out.qder  = qder;
out.qhigh = qhigh;

end

% ==========================================================
% Helpers (fast, allocation-light)
% ==========================================================
function T = expSE3_fast(Y, theta)
% Closed-form SE(3) exponential for Xi = [skew(w), v; 0 0]
w = Y(1:3); v = Y(4:6);
wn = norm(w);

if wn < 1e-12
    R = eye(3);
    p = v * theta;
else
    phi = wn * theta;
    u = w / wn;
    uhat = skew3(u);
    uhat2 = uhat*uhat;

    s = sin(phi);
    c = cos(phi);

    R = eye(3) + s*uhat + (1-c)*uhat2;

    What = skew3(w);
    What2 = What*What;

    A = (1 - c) / (wn^2);
    B = (phi - s) / (wn^3);

    V = eye(3)*theta + A*What + B*What2;
    p = V * v;
end

T = [R, p; 0 0 0 1];
end

function S = Ad_mul_Twist(T, Y)
% S = Ad(T) * Y without building 6x6 Ad
R = T(1:3,1:3);
p = T(1:3,4);
w = Y(1:3); v = Y(4:6);

Rw = R*w;
S  = [Rw;
      cross(p, Rw) + R*v];
end

function y = ad_mul_Twist(V, S)
% y = ad(V)*S without building 6x6
w  = V(1:3); v  = V(4:6);
Sw = S(1:3); Sv = S(4:6);
y  = [cross(w, Sw);
      cross(v, Sw) + cross(w, Sv)];
end

function y = minus_adT_mul(V, x)
% y = -ad(V)' * x without building 6x6
w  = V(1:3); v  = V(4:6);
xw = x(1:3); xv = x(4:6);
y  = [cross(w, xw) + cross(v, xv);
      cross(w, xv)];
end

function A = ad_mat(V)
w = V(1:3); v = V(4:6);
A = [skew3(w), zeros(3,3);
     skew3(v), skew3(w)];
end

function S = skew3(x)
S = [   0   -x(3)  x(2);
      x(3)    0   -x(1);
     -x(2)  x(1)    0 ];
end

function M0 = transformInertiaToWorld(C0, Mb)
% M0 = Ad(C0^{-1})' * Mb * Ad(C0^{-1})
R = C0(1:3,1:3);
p = C0(1:3,4);
Rinv = R.';
pinv = -Rinv*p;

Ad = [Rinv, zeros(3,3);
      skew3(pinv)*Rinv, Rinv];

M0 = Ad.' * Mb * Ad;
end

function x = solve_ldl_fact(L, D, perm, b)
bp = b(perm);
y  = L \ bp;
z  = D \ y;
xp = L.' \ z;

x = zeros(size(b));
x(perm) = xp;
end

% function out = GABI_FBS( ...
%     N, c, p, A0_1, Ap_j, C0_1, Y, Mb, V0_1, q0, qdot, ...
%     Wprop, tau_bar, Wapp, rmax, g)
% %FDYN_ABA_TREE_SPATIAL_PROGRESSIVE_SELFCONTAINED
% % Progressive higher-order ABA forward dynamics for floating-base trees.
% % Self-contained: computes S and its derivatives exactly like your FK recursion.
% %
% % Spatial convention:
% %   twist  V = [omega; v]
% %   wrench W = [t; f]
% %   all expressed in inertial frame {0}.
% %
% % INPUTS (keep at top):
% %   N        : number of bodies, base is body 1
% %   c        : 1xN cell, children list
% %   p        : parent array (p(1)=0)
% %   A0_1     : 4x4 constant home transform A^0_1
% %   Ap_j     : 1xN cell, Ap_j{j}=A^{p(j)}_j constant home transform (parent->child)
% %   C0_1     : 4x4 current base pose C^0_1(t)
% %   Y        : 1xN cell, constant spatial screws at home (in inertial), Y{1}=0
% %   Mb       : 1xN cell, body-fixed spatial inertia (6x6) constant
% %   V0_1     : 6x1 base twist V^0_1 (0th order)
% %   q0       : Nx1 joint positions (q0(1) unused)
% %   qdot     : Nx1 joint velocities (qdot(1) unused)
% %   Wprop    : 6x(rmax+1), (W_prop)^(r), r=0..rmax
% %   tau_bar  : Nx(rmax+1), tau^(r), r=0..rmax (tau_bar(1,:) unused)
% %   Wapp     : 1xN cell, each 6x(rmax+1), (W_app,j)^(r)
% %   rmax     : maximum order >= 0
% %   g        : gravity constant
% %
% % OUTPUTS:
% %   out.Vder{j} : 6x(rmax+2), V^(0..rmax+1)
% %   out.qder    : Nx(rmax+3), q^(0..rmax+2) (q^(2)=accel, q^(3)=jerk, ...)
% %   out.qhigh   : Nx(rmax+1), q^(r+2) for r=0..rmax
% %   out.MA{j}   : articulated inertia (computed once)
% %   out.WA{j}   : 6x(rmax+1) articulated bias wrenches WA^(r)
% %   out.Sder{j} : 6x(rmax+2) screw derivatives S^(0..rmax+1)
% 
% % ----------------------------
% % checks
% % ----------------------------
% assert(iscell(c) && numel(c)==N);
% assert(iscell(Ap_j) && numel(Ap_j)==N);
% assert(iscell(Y) && numel(Y)==N);
% assert(iscell(Mb) && numel(Mb)==N);
% assert(iscell(Wapp) && numel(Wapp)==N);
% assert(all(size(Wprop)==[6, rmax+1]));
% assert(all(size(tau_bar)==[N, rmax+1]));
% for j=1:N
%     assert(all(size(Mb{j})==[6 6]));
%     assert(all(size(Wapp{j})==[6, rmax+1]));
%     assert(all(size(Y{j})==[6 1]));
% end
% assert(all(size(C0_1)==[4 4]) && all(size(A0_1)==[4 4]));
% assert(numel(q0)==N && numel(qdot)==N);
% assert(all(size(V0_1)==[6 1]));
% 
% G0 = [0;0;0; 0;0;-g];
% 
% % ----------------------------
% % Step 1) Build constant A0_j and current F_j, C0_j (0th order position)
% %   F_1 = C0_1
% %   F_j = F_p exp([Y_j] q_j)
% %   A0_1 = A0_1, A0_j = A0_p * A^{p}_j
% %   C0_j = F_j * A0_j
% % ----------------------------
% A0 = cell(1,N);
% F  = cell(1,N);
% C0 = cell(1,N);
% 
% A0{1} = A0_1;
% F{1}  = C0_1;
% C0{1} = F{1} * A0{1};
% 
% ForwardPose(1);
% 
% % ----------------------------
% % Step 2) Compute S^(0) from S = Ad_{F_j} Y_j  (since C0_j * inv(A0_j) = F_j)
% % and compute V^(0) via velocity recursion V_j = V_p + S_j*qdot_j
% % ----------------------------
% Sder = cell(1,N);   % will hold S^(0..rmax+1)
% Vder = cell(1,N);   % will hold V^(0..rmax+1)
% for j=1:N
%     Sder{j} = zeros(6, rmax+2);  % need up to S^(rmax+1)
%     Vder{j} = zeros(6, rmax+2);  % need up to V^(rmax+1)
% end
% 
% Sder{1}(:,1) = zeros(6,1);
% Vder{1}(:,1) = V0_1;
% 
% ForwardSV0(1);
% 
% % ----------------------------
% % Step 3) Compute spatial inertia M0_j in inertial frame
% %   M0 = Ad_{C^{-1}}^T Mb Ad_{C^{-1}}
% % ----------------------------
% M0 = cell(1,N);
% for j=1:N
%     AdCinv = Ad_SE3(invSE3(C0{j}));
%     M0{j}  = AdCinv.' * Mb{j} * AdCinv;
% end
% 
% % ----------------------------
% % Step 4) Articulated inertia MA (order-independent), using S^(0)
% % ----------------------------
% MA  = cell(1,N);
% Dsc = nan(N,1); % D_j = S_j^T MA_j S_j
% Backward_MA(1);
% 
% % ----------------------------
% % Storage for progressive solve
% % ----------------------------
% qder = zeros(N, rmax+3);     % q^(0..rmax+2)
% qder(:,1) = q0(:);
% qder(:,2) = qdot(:);
% qder(1,:) = NaN;
% 
% WA = cell(1,N);
% for j=1:N
%     WA{j} = zeros(6, rmax+1); % WA^(0..rmax)
% end
% 
% qhigh = nan(N, rmax+1);      % q^(r+2)
% qtil  = nan(N, rmax+1);      % tilde q^(r+2)
% 
% % ----------------------------
% % MAIN progressive order loop
% % ----------------------------
% for r = 0:rmax
%     % Before solving order r, we must have S^(0..r+1).
%     % We already have S^(0). We build the next derivative layer by layer.
%     ensure_S_upto(r+1);
% 
%     % Backward for WA^(r) and qtil^(r+2)
%     Backward_WA_and_qtil(1, r);
% 
%     % Forward for base V^(r+1), then q^(r+2), and V^(r+1) down tree
%     Forward_order(1, r);
% 
%     % Pass forward q^(r+2) so next order can use it as "known"
%     qder(:, r+3) = qhigh(:, r+1);
% end
% 
% % Pack
% out.Vder = Vder;
% out.qder = qder;
% out.qhigh = qhigh;
% out.MA = MA;
% out.WA = WA;
% out.Sder = Sder;
% out.C0 = C0;
% out.F  = F;
% out.A0 = A0;
% 
% % ==========================================================
% % Nested: position recursion (pre-order)
% % ==========================================================
%     function ForwardPose(j)
%         kids = c{j};
%         for ii=1:numel(kids)
%             ch = kids(ii);
%             A0{ch} = A0{j} * Ap_j{ch};
%             F{ch}  = F{j} * expm_se3( vec_to_se3(Y{ch}) * q0(ch) );
%             C0{ch} = F{ch} * A0{ch};
%             ForwardPose(ch);
%         end
%     end
% 
% % ==========================================================
% % Nested: compute S^(0) and V^(0) (pre-order)
% % ==========================================================
%     function ForwardSV0(j)
%         kids = c{j};
%         for ii=1:numel(kids)
%             ch = kids(ii);
% 
%             % S^(0) = Ad_{F_ch} Y_ch
%             Sder{ch}(:,1) = Ad_SE3(F{ch}) * Y{ch};
% 
%             % V^(0)_ch = V^(0)_parent + S^(0)_ch * qdot_ch
%             Vder{ch}(:,1) = Vder{j}(:,1) + Sder{ch}(:,1) * qdot(ch);
% 
%             ForwardSV0(ch);
%         end
%     end
% 
% % ==========================================================
% % Nested: ensure S derivatives up to order kmax (kmax>=0)
% % Uses your FK recursion:
% %   S^(k) = sum_{l=0}^{k-1} C(k-1,l) ad_{V^(l)} S^(k-1-l)
% % ==========================================================
%     function ensure_S_upto(kmax)
%         % We build globally order-by-order: k=1..kmax.
%         for k = 1:kmax
%             % if already filled for base of recursion, skip if nonzero check is too risky,
%             % so we just recompute deterministically (cheap, O(N*k)).
%             compute_S_order_k(1, k);
%         end
%     end
% 
%     function compute_S_order_k(j, k)
%         kids = c{j};
%         for ii=1:numel(kids)
%             ch = kids(ii);
% 
%             % Compute S_ch^(k)
%             Sk = zeros(6,1);
%             for l = 0:(k-1)
%                 coeff = nchoosek(k-1, l);
%                 adVl  = ad_twist(Vder{ch}(:, l+1));         % V^(l)
%                 Sprev = Sder{ch}(:, (k-1-l)+1);            % S^(k-1-l)
%                 Sk = Sk + coeff * (adVl * Sprev);
%             end
%             Sder{ch}(:, k+1) = Sk;
% 
%             compute_S_order_k(ch, k);
%         end
%     end
% 
% % ==========================================================
% % Backward articulated inertia MA (post-order), uses S^(0)
% % ==========================================================
%     function Backward_MA(j)
%         kids = c{j};
%         for ii=1:numel(kids)
%             Backward_MA(kids(ii));
%         end
%         MAj = M0{j};
%         for ii=1:numel(kids)
%             i  = kids(ii);
%             Si = Sder{i}(:,1);           % S^(0)
%             MiA = MA{i};
%             Ui = MiA*Si;
%             Di = Si.'*Ui;
%             MAj = MAj + (MiA - (Ui*(1/Di))*Ui.');
%             Dsc(i) = Di;
%         end
%         MA{j} = MAj;
%     end
% 
% % ==========================================================
% % Backward WA^(r) and qtil^(r+2) (post-order)
% % ==========================================================
%     function Backward_WA_and_qtil(j, r)
%         kids = c{j};
%         for ii=1:numel(kids)
%             Backward_WA_and_qtil(kids(ii), r);
%         end
% 
%         V0j = Vder{j}(:,1); % V^(0)
% 
%         if r == 0
%             % WA^(0) for isolated body
%             WAj = - ad_twist(V0j).'*(M0{j}*V0j) ...
%                   - Wapp{j}(:,1) ...
%                   - (M0{j}*G0);
% 
%             % Add children and compute ddot_tilde
%             for ii=1:numel(kids)
%                 i   = kids(ii);
%                 Si  = Sder{i}(:,1);
%                 Sdi = Sder{i}(:,2); % S^(1)
%                 MiA = MA{i};
%                 WiA = WA{i}(:,1);
%                 Di  = Dsc(i);
%                 qd  = qder(i,2);
% 
%                 qtil(i,1) = (1/Di) * ( tau_bar(i,1) - Si.'*( MiA*(Sdi*qd) + WiA ) );
% 
%                 WAj = WAj + ( WiA + MiA*(Si*qtil(i,1) + Sdi*qd) );
%             end
% 
%             WA{j}(:,1) = WAj;
% 
%         else
%             % r>0: bias momentum term Pi^(r+1) with V^(r+1) set to 0
%             Pi_bias = compute_Pi_bias_rp1(j, r);
% 
%             % gravity derivative term uses M^(r)
%             Mr = compute_M_derivative(j, r);
% 
%             WAj_r = - Wapp{j}(:,r+1) - (Mr*G0) + Pi_bias;
% 
%             for ii=1:numel(kids)
%                 i   = kids(ii);
%                 MiA = MA{i};
%                 Si  = Sder{i}(:,1);
%                 Di  = Dsc(i);
% 
%                 % Vbias_i = (V_i)^(r+1) with parent (V_p)^(r+1)=0 and q^(r+2)=0
%                 Vbias_i = compute_Vbias_rp1(i, r);
% 
% % tau_tilde^(r)
% tau_tilde = 0;
% for k = 0:(r-1)
%     coeff = nchoosek(r,k);                 % <-- FIX (was r-1)
%     SderT = Sder{i}(:, (r-k)+1).';         % (S^T)^(r-k)
%     Vk1   = Vder{i}(:, (k+1)+1);           % V^(k+1)
%     WAk   = WA{i}(:, k+1);                 % WA^(k)
%     tau_tilde = tau_tilde + coeff*( SderT*(MiA*Vk1 + WAk) );
% end
% 
%                 % qtil^(r+2)
%                 qtil(i,r+1) = (1/Di) * ( ...
%                     tau_bar(i,r+1) - tau_tilde ...
%                     - Si.'*( MiA*Vbias_i + WA{i}(:,r+1) ) );
% 
%                 WAj_r = WAj_r + ( WA{i}(:,r+1) + MiA*(Si*qtil(i,r+1) + Vbias_i) );
%             end
% 
%             WA{j}(:,r+1) = WAj_r;
%         end
%     end
% 
% % ==========================================================
% % Forward pass for order r (pre-order)
% % ==========================================================
%     function Forward_order(j, r)
%         if j == 1
%             rhs = Wprop(:,r+1) - WA{1}(:,r+1);
%             Vder{1}(:, (r+1)+1) = solve_spd_ldl(MA{1}, rhs); % V^(r+1)
%         end
% 
%         kids = c{j};
%         for ii=1:numel(kids)
%             ch = kids(ii);
% 
%             Vp_rp1 = Vder{j}(:, (r+1)+1);
%             Sj     = Sder{ch}(:,1);
%             Di     = Dsc(ch);
% 
%             qhigh(ch,r+1) = -(1/Di) * ( Sj.'*(MA{ch}*Vp_rp1) ) + qtil(ch,r+1);
% 
%             Vbias_local = compute_Vbias_rp1(ch, r);
%             Vder{ch}(:, (r+1)+1) = Vp_rp1 + Vbias_local + Sj*qhigh(ch,r+1);
% 
%             Forward_order(ch, r);
%         end
%     end
% 
% % ==========================================================
% % Vbias: (V_j)^(r+1) with parent V^(r+1)=0 and q^(r+2)=0
% % From kinematics:
% %   V^(r+1) = Vp^(r+1) + sum_{a=0}^{r+1} C(r+1,a) S^(a) q^(r-a+2)
% % set Vp^(r+1)=0 and q^(r+2)=0 -> omit a=0 term
% % ==========================================================
%     function Vbias = compute_Vbias_rp1(j, r)
%         r1 = r+1;
%         Vbias = zeros(6,1);
%         for a = 1:r1
%             coeff = nchoosek(r1,a);
%             Sdera = Sder{j}(:, a+1);       % S^(a)
%             qord  = (r - a + 2);           % q^(r-a+2)
%             qterm = qder(j, qord+1);
%             Vbias = Vbias + coeff*(Sdera*qterm);
%         end
%     end
% 
% % ==========================================================
% % M^(r) using your recurrence with V^(0..r-1)
% % ==========================================================
%     function Mr = compute_M_derivative(j, r)
%         if r==0, Mr = M0{j}; return; end
%         Mloc = cell(r+1,1); Mloc{1}=M0{j};
%         for rr=1:r
%             Mrr = zeros(6,6);
%             for k=0:(rr-1)
%                 coeff = nchoosek(rr-1,k);
%                 Mprev = Mloc{(rr-1-k)+1};
%                 Vk    = Vder{j}(:, k+1);
%                 adVk  = ad_twist(Vk);
%                 Mrr = Mrr - coeff*(Mprev*adVk + adVk.'*Mprev);
%             end
%             Mloc{rr+1} = Mrr;
%         end
%         Mr = Mloc{r+1};
%     end
% 
% % ==========================================================
% % Pi^(r+1) bias: omit k=r+1 term (V^(r+1)=0)
% % Pi^(m)=sum_{k=0}^m C(m,k) M^(m-k) V^(k)
% % ==========================================================
%     function Pi_bias = compute_Pi_bias_rp1(j, r)
%         m = r+1;
%         % build M^(0..m)
%         Mloc = cell(m+1,1); Mloc{1}=M0{j};
%         for rr=1:m
%             Mrr = zeros(6,6);
%             for k=0:(rr-1)
%                 coeff = nchoosek(rr-1,k);
%                 Mprev = Mloc{(rr-1-k)+1};
%                 Vk    = Vder{j}(:, k+1);
%                 adVk  = ad_twist(Vk);
%                 Mrr = Mrr - coeff*(Mprev*adVk + adVk.'*Mprev);
%             end
%             Mloc{rr+1} = Mrr;
%         end
%         Pi_bias = zeros(6,1);
%         for k=0:r
%             coeff = nchoosek(m,k);
%             Pi_bias = Pi_bias + coeff*( Mloc{(m-k)+1} * Vder{j}(:,k+1) );
%         end
%     end
% 
% end
% 
% % =========================
% % Helpers
% % =========================
% function Tinv = invSE3(T)
% R=T(1:3,1:3); p=T(1:3,4);
% Tinv=[R.', -R.'*p; 0 0 0 1];
% end
% 
% function A = Ad_SE3(T)
% R=T(1:3,1:3); p=T(1:3,4);
% A=[R, zeros(3,3); skew3(p)*R, R];
% end
% 
% function a = ad_twist(V)
% w=V(1:3); v=V(4:6);
% a=[skew3(w), zeros(3,3); skew3(v), skew3(w)];
% end
% 
% function S = skew3(x)
% S=[0 -x(3) x(2); x(3) 0 -x(1); -x(2) x(1) 0];
% end
% 
% function x = solve_spd_ldl(A,b)
% [L,D,perm]=ldl(A,'vector');
% bp=b(perm); y=L\bp; z=D\y; xp=L.'\z;
% x=zeros(size(b)); x(perm)=xp;
% end
% 
% function Xi = vec_to_se3(Y)
% % Y = [w; v], se(3) matrix
% w = Y(1:3); v = Y(4:6);
% Xi = [skew3(w), v; 0 0 0 0];
% end
% 
% function T = expm_se3(Xi)
% % use expm for robustness (you can replace with closed-form if you want)
% T = expm(Xi);
% end