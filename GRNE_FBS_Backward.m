function out = GRNE_FBS_Backward(N, c, p, fk, Mb, Wapp0, r, g)
% Optimized higher-order backward inverse dynamics (spatial) for floating-base trees.
% Spatial convention: twist [omega; v], wrench [n; f], expressed in inertial {0}.

DEBUG        = false;   % slower if true
RETURN_CELLS = true;    % set false for fastest + lowest memory

% ----------------------------
% checks (optional)
% ----------------------------
if DEBUG
    assert(iscell(c) && numel(c)==N);
    assert(iscell(Mb) && numel(Mb)==N);
    assert(iscell(Wapp0) && numel(Wapp0)==N);
end

% ----------------------------
% Precompute binomials B(n+1,k+1)=C(n,k), up to n=r+1
% ----------------------------
nMax = r + 1;
B = zeros(nMax+2, nMax+2);
B(1,1) = 1;
for n = 1:(nMax+1)
    B(n+1,1)   = 1;
    B(n+1,n+1) = 1;
    for k = 1:(n-1)
        B(n+1,k+1) = B(n,k+1) + B(n,k);
    end
end

% ----------------------------
% Build postorder traversal (iterative)
% ----------------------------
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
for i = 1:sp2
    postOrder(i) = stk2(sp2 - i + 1);
end
postOrder = postOrder(1:sp2);

% ----------------------------
% Convert fk fields to arrays (once)
% Supports either cell-style fk (your original) or array-style (from fast forward)
% ----------------------------
C0 = zeros(4,4,N);
V  = zeros(6, r+2, N);     % need up to r+1 => r+2 cols
S  = zeros(6, r+1, N);     % need up to r

if iscell(fk.C0)
    for j = 1:N
        C0(:,:,j) = fk.C0{j};
        V(:,:,j)  = fk.V{j}(:,1:(r+2));
        S(:,:,j)  = fk.S{j}(:,1:(r+1));
    end
else
    C0 = fk.C0;
    V  = fk.V(:,1:(r+2),:);
    S  = fk.S(:,1:(r+1),:);
end

% Convert Wapp0 to array for faster access
Wapp = zeros(6, r+1, N);
for j = 1:N
    Wapp(:,:,j) = Wapp0{j};
end

% Gravity spatial acceleration vector
G0 = [0;0;0; 0;0;-g];

% ----------------------------
% Allocate outputs (numeric, contiguous)
% ----------------------------
W     = zeros(6, r+1, N);         % constraint wrench derivatives
Wgrav = zeros(6, r+1, N);         % gravity wrench derivatives
Pi    = zeros(6, r+2, N);         % momentum derivatives 0..r+1
Mder  = zeros(6, 6, N, r+2);      % inertia derivatives 0..r+1

% Q: store as 6×(r+1)×N internally.
% - for base (j=1): full 6×(r+1)
% - for joints (j>1): only Q(1,:,j) used
Qfull = zeros(6, r+1, N);

% Optional cache: ad(V^(k)) for k=0..r (needed for M^(1..r+1))
adV = zeros(6,6,N, r+1);
for j = 1:N
    for k = 0:r
        adV(:,:,j,k+1) = ad_mat(V(:,k+1,j));
    end
end

% ----------------------------
% STEP 1: per-body M-derivs, Pi-derivs, Wgrav
% ----------------------------
for j = 1:N
    % M^(0) from Mb via Ad(C^{-1})' Mb Ad(C^{-1})
    M0 = transformInertiaToWorld(C0(:,:,j), Mb{j});
    Mder(:,:,j,1) = M0;

    % M^(rr), rr=1..r+1
    for rr = 1:(r+1)
        Mrr = zeros(6,6);
        % M^(rr) = - sum_{k=0}^{rr-1} C(rr-1,k) [ M^(rr-1-k) ad(V^(k)) + ad(V^(k))' M^(rr-1-k) ]
        for k = 0:(rr-1)
            coeff = B(rr, k+1);                       % C(rr-1,k)
            Mprev = Mder(:,:,j, rr-k);                % (rr-1-k)+1
            A     = adV(:,:,j, k+1);                  % ad(V^(k))
            Mrr   = Mrr - coeff*(Mprev*A + A.'*Mprev);
        end
        Mder(:,:,j, rr+1) = Mrr;
    end

    % Pi^(m), m=0..r+1 : Pi^(m) = sum_{k=0}^m C(m,k) M^(m-k) V^(k)
    for m = 0:(r+1)
        Pim = zeros(6,1);
        for k = 0:m
            coeff = B(m+1, k+1);                      % C(m,k)
            Mk    = Mder(:,:,j, (m-k)+1);
            Pim   = Pim + coeff*(Mk * V(:,k+1,j));
        end
        Pi(:,m+1,j) = Pim;
    end

    % Wgrav^(k) = M^(k) * G0 for k=0..r
    for k = 0:r
        Wgrav(:,k+1,j) = Mder(:,:,j,k+1) * G0;
    end
end

% ----------------------------
% STEP 2: backward accumulation in postorder
% ----------------------------
for idx = 1:numel(postOrder)
    j = postOrder(idx);
    kids = c{j};

    % sum children wrenches (all orders at once)
    sumChild = zeros(6, r+1);
    for ii = 1:numel(kids)
        sumChild = sumChild + W(:,:,kids(ii));
    end

    % W(:,:,j) = Pi(:,2:r+2,j) - Wapp(:,:,j) - Wgrav(:,:,j) + sumChild
    W(:,:,j) = Pi(:, 2:(r+2), j) - Wapp(:,:,j) - Wgrav(:,:,j) + sumChild;

    % Project to generalized forces
    if j == 1
        Qfull(:,:,1) = W(:,:,1);    % base wrench required (prop if Wapp0 excluded it)
    else
        % Q^(0) = S^(0)^T W^(0)
        Qfull(1,1,j) = S(:,1,j).'*W(:,1,j);

        % Q^(rr) = sum_{k=0}^{rr} C(rr,k) (S^T)^(rr-k) W^(k)
        for rr = 1:r
            qrr = 0;
            for k = 0:rr
                coeff = B(rr+1, k+1);                    % C(rr,k)
                qrr   = qrr + coeff*( S(:, (rr-k)+1, j).' * W(:, k+1, j) );
            end
            Qfull(1, rr+1, j) = qrr;
        end
    end
end

% ----------------------------
% Pack outputs
% ----------------------------
if RETURN_CELLS
    out.W     = cell(1,N);
    out.Q     = cell(1,N);
    out.M0    = cell(1,N);
    out.Pi    = cell(1,N);
    out.Wgrav = cell(1,N);

    for j = 1:N
        out.W{j}     = W(:,:,j);
        out.Wgrav{j} = Wgrav(:,:,j);
        out.M0{j}    = squeeze(Mder(:,:,j,:));      % 6x6x(r+2)
        out.Pi{j}    = Pi(:,:,j);                   % 6x(r+2)
        if j == 1
            out.Q{j} = Qfull(:,:,j);                % 6x(r+1)
        else
            out.Q{j} = Qfull(1,:,j);                % 1x(r+1)
        end
    end
else
    out.W     = W;
    out.Wgrav = Wgrav;
    out.M0    = Mder;
    out.Pi    = Pi;
    out.Q     = Qfull; % interpret as: base full, joints in row 1
end

end

% ==========================================================
% Helpers (fast)
% ==========================================================
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

function A = ad_mat(V)
w = V(1:3); v = V(4:6);
A = [skew3(w), zeros(3,3);
     skew3(v), skew3(w)];
end

function S = skew3(x)
S = [  0   -x(3)  x(2);
      x(3)  0    -x(1);
     -x(2) x(1)   0  ];
end

% function out = GRNE_FBS_Backward( ...
%     N, c, p, fk, Mb, Wapp0, r, g)
% %IDYN_TREE_HIGHER_ORDER Higher-order recursive inverse dynamics (spatial) for floating-base trees.
% %
% % Spatial convention: twist/wrench are 6x1 with [omega; v] and [n; f] (angular/torque part first).
% % All spatial quantities are expressed in the inertial frame {0}.
% %
% % ----------------------------
% % USER INPUTS (KEEP HERE)
% % ----------------------------
% % N      : number of bodies, indexed 1..N (1 = floating base)
% % c      : 1xN cell, c{j} = array of children indices of body j
% % p      : parent index array, p(1)=0 (unused), p(j) in 1..N for j>1
% % fk     : struct from forward recursion with at least:
% %          fk.C0{j} : 4x4 pose C^0_j
% %          fk.V{j}  : 6x(r+2) twist derivatives, fk.V{j}(:,k+1) = (V^0_j)^{(k)}, k=0..r+1
% %          fk.S{j}  : 6x(r+1) screw derivatives, fk.S{j}(:,k+1) = S_j^{(k)}, k=0..r
% % Mb     : 1xN cell, Mb{j} is 6x6 constant body-fixed spatial inertia matrix M^b_j
% % Wapp0  : 1xN cell, Wapp0{j} is 6x(r+1) applied spatial wrench derivatives at body j:
% %          Wapp0{j}(:,k+1) = (W^0_{j,app})^{(k)}, k=0..r
% %          IMPORTANT: this must EXCLUDE the propeller wrench at the base (j=1) if you want Q1=prop.
% % r      : highest derivative order of inverse dynamics wrench/torque to compute (>=0)
% % g      : gravity acceleration constant (e.g., 9.81)
% %
% % ----------------------------
% % OUTPUT
% % ----------------------------
% % out.W{j}     : 6x(r+1) constraint wrench derivatives transmitted from body j to its parent:
% %               out.W{j}(:,k+1) = (W^0_j)^{(k)}
% % out.Q{j}     : generalized force derivatives:
% %               - if j==1: out.Q{1} is 6x(r+1) (base wrench = required prop wrench)
% %               - if j>1 : out.Q{j} is 1x(r+1) (joint torque/force and its derivatives)
% % out.M0{j}    : 6x6x(r+2) spatial inertia derivatives M^0_j^{(k)}, k=0..r+1
% % out.Pi{j}    : 6x(r+2) momentum derivatives Pi^{(k)}, k=0..r+1
% % out.Wgrav{j} : 6x(r+1) gravity wrench derivatives
% %
% % ----------------------------
% % KEY REQUIREMENT
% % ----------------------------
% % For inverse dynamics of order r, you NEED fk.V up to order r+1 (so size 6x(r+2)).
% % Because (W^0)^{(r)} uses Pi^{(r+1)}, which depends on V^{(r+1)} and M^{(r+1)}.
% 
% % ----------------------------
% % BASIC CHECKS
% % ----------------------------
% assert(iscell(c) && numel(c)==N, 'c must be 1xN cell.');
% assert(iscell(Mb) && numel(Mb)==N, 'Mb must be 1xN cell.');
% assert(iscell(Wapp0) && numel(Wapp0)==N, 'Wapp0 must be 1xN cell.');
% for j = 1:N
%     assert(all(size(Mb{j})==[6 6]), 'Mb{%d} must be 6x6.', j);
%     assert(size(Wapp0{j},1)==6 && size(Wapp0{j},2)==r+1, 'Wapp0{%d} must be 6x(r+1).', j);
%     assert(size(fk.V{j},1)==6 && size(fk.V{j},2)>=r+2, 'fk.V{%d} must be 6x(r+2) or larger.', j);
%     assert(size(fk.S{j},1)==6 && size(fk.S{j},2)>=r+1, 'fk.S{%d} must be 6x(r+1) or larger.', j);
% end
% 
% % ----------------------------
% % PRECOMPUTE BINOMIAL COEFS
% % ----------------------------
% % binom{n+1}(k+1) = nchoosek(n,k)
% maxN = r+1; % we will need up to (r+1 choose k) for M^{(r+1)} and Pi^{(r+1)}
% binom = cell(maxN+1,1);
% for n = 0:maxN
%     b = zeros(1,n+1);
%     for k = 0:n
%         b(k+1) = nchoosek(n,k);
%     end
%     binom{n+1} = b;
% end
% 
% % ----------------------------
% % ALLOCATE OUTPUTS
% % ----------------------------
% W      = cell(1,N);   % 6x(r+1)
% Q      = cell(1,N);   % base: 6x(r+1), others: 1x(r+1)
% M0der  = cell(1,N);   % 6x6x(r+2) for k=0..r+1
% Pider  = cell(1,N);   % 6x(r+2) for k=0..r+1
% Wgrav  = cell(1,N);   % 6x(r+1)
% 
% for j = 1:N
%     W{j}     = zeros(6, r+1);
%     Wgrav{j} = zeros(6, r+1);
%     M0der{j} = zeros(6, 6, r+2);  % store M^{(0..r+1)}
%     Pider{j} = zeros(6, r+2);     % store Pi^{(0..r+1)}
%     if j == 1
%         Q{j} = zeros(6, r+1);
%     else
%         Q{j} = zeros(1, r+1);
%     end
% end
% 
% % Gravity spatial acceleration vector (twist-acc style you used): [0;0;0; 0;0;-g]
% G0 = [0;0;0; 0;0;-g];
% 
% % ----------------------------
% % STEP 1: PER-BODY INERTIA DERIVS, MOMENTUM DERIVS, GRAVITY DERIVS
% % (independent of tree backward accumulation)
% % ----------------------------
% for j = 1:N
%     % ---- M^0_j (0th) from Mb via Ad_{C}^{-T} Mb Ad_{C}^{-1}
%     C0 = fk.C0{j};
%     AdCinv = Ad_SE3(invSE3(C0));        % Ad_{C^{-1}}
%     M0der{j}(:,:,1) = AdCinv.' * Mb{j} * AdCinv;   % M^{(0)}
% 
%     % ---- M^{(k)} for k=1..r+1 using your recursion (Mass)
%     % M^{(r)} = - sum_{k=0}^{r-1} C(r-1,k) [ M^{(r-1-k)} ad_{V^{(k)}} + ad_{V^{(k)}}^T M^{(r-1-k)} ]
%     for rr = 1:(r+1)
%         Mr = zeros(6,6);
%         b = binom{rr}; % corresponds to (rr-1 choose k), length rr
%         for k = 0:(rr-1)
%             coeff = b(k+1);
%             Mprev = M0der{j}(:,:, (rr-1-k)+1); % M^{(rr-1-k)}
%             Vk    = fk.V{j}(:, k+1);           % V^{(k)}
%             adVk  = ad_twist(Vk);
%             Mr = Mr - coeff * ( Mprev*adVk + adVk.'*Mprev );
%         end
%         M0der{j}(:,:, rr+1) = Mr; % store M^{(rr)}
%     end
% 
%     % ---- Pi^{(m)} for m=0..r+1 using Momu
%     % Pi^{(m)} = sum_{k=0}^{m} C(m,k) M^{(m-k)} V^{(k)}
%     for m = 0:(r+1)
%         Pim = zeros(6,1);
%         bm = binom{m+1}; % (m choose k), length m+1
%         for k = 0:m
%             coeff = bm(k+1);
%             Mmk = M0der{j}(:,:, (m-k)+1); % M^{(m-k)}
%             Vk  = fk.V{j}(:, k+1);        % V^{(k)}
%             Pim = Pim + coeff * (Mmk * Vk);
%         end
%         Pider{j}(:, m+1) = Pim;
%     end
% 
%     % ---- Gravity wrench derivatives: Wgrav^{(k)} = M^{(k)} * G0, k=0..r
%     for k = 0:r
%         Wgrav{j}(:, k+1) = M0der{j}(:,:, k+1) * G0;
%     end
% end
% 
% % ----------------------------
% % STEP 2: BACKWARD (POST-ORDER DFS) TO ACCUMULATE CONSTRAINT WRENCHES
% % ----------------------------
% Backward(1);
% 
% % Pack outputs
% out.W     = W;
% out.Q     = Q;
% out.M0    = M0der;
% out.Pi    = Pider;
% out.Wgrav = Wgrav;
% 
% % ==========================================================
% % NESTED FUNCTION: BACKWARD RECURSION (POST-ORDER DFS)
% % ==========================================================
%     function Backward(j)
%         % First recurse into children
%         kids = c{j};
%         for ii = 1:numel(kids)
%             Backward(kids(ii));
%         end
% 
%         % Sum children constraint wrenches (order-wise)
%         sumChild = zeros(6, r+1);
%         for ii = 1:numel(kids)
%             sumChild = sumChild + W{kids(ii)};
%         end
% 
%         % ---- Compute constraint wrench derivatives W_j^{(k)} for k=0..r
%         % From your Algorithm (Dyn):
%         % 0th:   W_j = Pi_dot - Wapp - Wgrav + sum_child W_i
%         % rth: (W_j)^{(r)} = Pi^{(r+1)} - Wapp^{(r)} - Wgrav^{(r)} + sum_child (W_i)^{(r)}
%         %
%         % Note: Pi_dot = Pi^{(1)}. Generally, for order k: use Pi^{(k+1)}.
%         for k = 0:r
%             Pi_kp1 = Pider{j}(:, (k+1)+1); % Pi^{(k+1)}
%             W{j}(:, k+1) = Pi_kp1 ...
%                 - Wapp0{j}(:, k+1) ...
%                 - Wgrav{j}(:, k+1) ...
%                 + sumChild(:, k+1);
%         end
% 
%         % ---- Project to generalized forces
%         if j == 1
%             % Floating base: generalized "force" is the full base wrench.
%             % If Wapp0{1} excluded the prop wrench, then Q1 equals required prop wrench.
%             Q{1} = W{1};
%         else
%             % Joint torque/force:
%             % Q^{(0)} = S^T W
%             Q{j}(1) = fk.S{j}(:,1).'*W{j}(:,1);
% 
%             % Higher derivatives:
%             % (Q)^{(r)} = sum_{k=0}^{r} C(r,k) (S^T)^{(r-k)} (W)^{(k)}
%             for rr = 1:r
%                 qrr = 0;
%                 b = binom{rr+1}; % (rr choose k)
%                 for k = 0:rr
%                     coeff = b(k+1);
%                     SderT = fk.S{j}(:, (rr-k)+1).'; % (S^T)^{(rr-k)}
%                     Wder  = W{j}(:, k+1);           % W^{(k)}
%                     qrr = qrr + coeff * (SderT * Wder);
%                 end
%                 Q{j}(rr+1) = qrr;
%             end
%         end
%     end
% 
% end % idyn_tree_higher_order
% 
% % ==========================================================
% % HELPER FUNCTIONS (LOCAL)
% % ==========================================================
% function Tinv = invSE3(T)
% R = T(1:3,1:3);
% p = T(1:3,4);
% Tinv = [R.', -R.'*p; 0 0 0 1];
% end
% 
% function A = Ad_SE3(T)
% % Adjoint map Ad_T for SE(3), with twist [omega; v]
% R = T(1:3,1:3);
% p = T(1:3,4);
% A = [R, zeros(3,3);
%      skew3(p)*R, R];
% end
% 
% function a = ad_twist(V)
% % Lie algebra adjoint ad_V such that ad_V * W = V x W for twists [omega; v]
% w = V(1:3);
% v = V(4:6);
% a = [skew3(w), zeros(3,3);
%      skew3(v), skew3(w)];
% end
% 
% function S = skew3(x)
% S = [  0   -x(3)  x(2);
%       x(3)  0    -x(1);
%      -x(2) x(1)   0  ];
% end