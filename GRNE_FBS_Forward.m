function out = GRNE_FBS_Forward( ...
    N, c, p, A0_1, Ap_j, C0_1, V0_1_derivs, q_derivs, Y, r)

% Optimized higher-order forward kinematics (spatial) for floating-base trees:
% - iterative preorder traversal (no recursion)
% - Pascal triangle binomials (no nchoosek)
% - closed-form SE(3) exp (no expm)
% - Ad(T)*Y and ad(V)*S computed without building 6x6 matrices
% - contiguous arrays: F/A0/C0 are 4x4xN, S/V are 6x(r+1)xN

DEBUG        = false;   % set true for asserts (slower)
RETURN_CELLS = true;    % set false for fastest + lowest memory

if DEBUG
    assert(iscell(c) && numel(c)==N);
    assert(iscell(Ap_j) && numel(Ap_j)==N);
    assert(iscell(Y) && numel(Y)==N);
    assert(all(size(C0_1)==[4 4]));
    assert(all(size(A0_1)==[4 4]));
    assert(all(size(V0_1_derivs)==[6 r+1]));
    assert(size(q_derivs,1)==N && size(q_derivs,2)>=r+2);
end

% -----------------------
% Precompute binomials B(n+1,k+1)=C(n,k), up to n=r
% -----------------------
B = zeros(r+2, r+2);
B(1,1) = 1;
for n = 1:(r+1)
    B(n+1,1)   = 1;
    B(n+1,n+1) = 1;
    for k = 1:(n-1)
        B(n+1,k+1) = B(n,k+1) + B(n,k);
    end
end

% -----------------------
% Build iterative preorder traversal
% -----------------------
preOrder = zeros(N,1);
stk = zeros(N,1); sp = 1; stk(sp) = 1; m = 0;
while sp > 0
    j = stk(sp); sp = sp - 1;
    m = m + 1; preOrder(m) = j;
    kids = c{j};
    for ii = numel(kids):-1:1
        sp = sp + 1; stk(sp) = kids(ii);
    end
end
preOrder = preOrder(1:m);

% -----------------------
% Allocate contiguous storage
% -----------------------
F  = zeros(4,4,N);
A0 = zeros(4,4,N);
C0 = zeros(4,4,N);

S  = zeros(6, r+1, N);
V  = zeros(6, r+1, N);

% -----------------------
% Initialize base (j=1)
% -----------------------
F(:,:,1)  = C0_1;
A0(:,:,1) = A0_1;
C0(:,:,1) = C0_1;          % matches your original behavior
V(:,:,1)  = V0_1_derivs;   % V0_1_derivs(:,k+1) = (V^0_1)^(k)

% -----------------------
% Preorder forward pass
% -----------------------
for idx = 2:numel(preOrder)
    j  = preOrder(idx);
    pj = p(j);
    if DEBUG, assert(pj>=1 && pj<=N); end

    % ---- 0th order pose recursion
    q0j      = q_derivs(j,1);                       % q^(0)
    F(:,:,j) = F(:,:,pj) * expSE3_fast(Y{j}, q0j);  % F_j = F_p * exp([Y] q)

    A0(:,:,j) = A0(:,:,pj) * Ap_j{j};               % constant
    C0(:,:,j) = F(:,:,j) * A0(:,:,j);               % world pose

    % ---- S^(0) = Ad(F_j) * Y_j (without 6x6 Ad)
    S(:,1,j) = Ad_mul_Twist(F(:,:,j), Y{j});

    % ---- V^(0) = Vp^(0) + S^(0) qdot
    qdot = q_derivs(j,2);                           % q^(1)
    V(:,1,j) = V(:,1,pj) + S(:,1,j) * qdot;

    % ---- Higher derivatives k = 1..r
    % S^(k) and V^(k) computed in increasing k, so dependencies are available.
    for k = 1:r
        % S^(k) = sum_{r2=0}^{k-1} C(k-1,r2) ad_{V^(r2)} S^(k-1-r2)
        Sk = zeros(6,1);
        for r2 = 0:(k-1)
            coeff = B(k, r2+1);                 % C(k-1,r2)
            V_r2  = V(:, r2+1, j);              % V^(r2)
            Sprev = S(:, k-r2, j);              % S^(k-1-r2) at col (k-1-r2)+1 = k-r2
            Sk    = Sk + coeff * ad_mul_Twist(V_r2, Sprev);
        end
        S(:, k+1, j) = Sk;

        % V^(k) = Vp^(k) + sum_{r2=0}^{k} C(k,r2) S^(r2) q^(k-r2+1)
        Vk = V(:, k+1, pj);
        for r2 = 0:k
            coeff = B(k+1, r2+1);               % C(k,r2)
            qord  = k - r2 + 1;                 % q^(qord)
            qterm = q_derivs(j, qord+1);        % +1 shift
            Vk    = Vk + coeff * ( S(:, r2+1, j) * qterm );
        end
        V(:, k+1, j) = Vk;
    end
end

% -----------------------
% Pack outputs
% -----------------------
if RETURN_CELLS
    out.F  = cell(1,N);
    out.A0 = cell(1,N);
    out.C0 = cell(1,N);
    out.S  = cell(1,N);
    out.V  = cell(1,N);
    for j = 1:N
        out.F{j}  = F(:,:,j);
        out.A0{j} = A0(:,:,j);
        out.C0{j} = C0(:,:,j);
        out.S{j}  = S(:,:,j);
        out.V{j}  = V(:,:,j);
    end
else
    out.F  = F;
    out.A0 = A0;
    out.C0 = C0;
    out.S  = S;
    out.V  = V;
end

end

% ==========================================================
% Helpers (fast, allocation-light)
% ==========================================================
function T = expSE3_fast(Y, theta)
% Closed-form SE(3) exponential for exp([Y] theta), Y=[w;v] (not necessarily unit w)
w = Y(1:3); v = Y(4:6);
wn = norm(w);

if wn < 1e-12
    R = eye(3);
    p = v * theta;
else
    phi = wn * theta;
    c = cos(phi);
    s = sin(phi);

    What  = skew3(w);
    What2 = What*What;

    % Rodrigues for R
    R = eye(3) + (s/wn)*What + ((1-c)/(wn^2))*What2;

    % V matrix for translation p = V*v
    A = (1 - c)/(wn^2);
    B = (phi - s)/(wn^3);
    V = eye(3)*theta + A*What + B*What2;

    p = V * v;
end

T = [R, p; 0 0 0 1];
end

function S = Ad_mul_Twist(T, Y)
% S = Ad(T)*Y without building 6x6 Ad
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

function S = skew3(x)
S = [   0   -x(3)  x(2);
      x(3)    0   -x(1);
     -x(2)  x(1)    0 ];
end

% function out = GRNE_FBS_Forward( ...
%     N, c, p, A0_1, Ap_j, C0_1, V0_1_derivs, q_derivs, Y, r)
% %FKINE_TREE_HIGHER_ORDER Higher-order recursive forward kinematics (spatial) for floating-base trees.
% %
% % Spatial convention (as you stated): twist is [omega; v] with angular first.
% % All twists/screws are expressed in the inertial frame {0} (spatial / right-invariant form).
% %
% % ----------------------------
% % USER INPUTS (KEEP HERE)
% % ----------------------------
% % N              : number of bodies, indexed 1..N (body 1 is floating base)
% % c              : 1xN cell, c{j} = array of children indices of body j
% % p              : 1xN (or Nx1) parent index array; p(1) can be 0 or 1 (unused)
% % A0_1           : 4x4 constant transform A^0_1 at home (often eye(4))
% % Ap_j           : 1xN cell, Ap_j{j} = A^{p{j}}_j (4x4) for j>1; Ap_j{1} ignored
% % C0_1           : 4x4 base pose C^0_1(t) in SE(3)
% % V0_1_derivs    : 6x(r+1) base spatial twist derivatives:
% %                  V0_1_derivs(:,k+1) = (V^0_1)^{(k)} for k=0..r
% % q_derivs       : Nx(r+2) joint derivatives for each body index j:
% %                  q_derivs(j,k+1) = q_j^{(k)} for k=0..(r+1)
% %                  NOTE: To compute (V_j)^{(r)} you need q^{(r+1)} (hence r+2 columns).
% %                  For body 1 (base), row is ignored (can be zeros).
% % Y              : 1xN cell, Y{j} is 6x1 constant screw axis for joint j (j>1), at home
% % r              : highest derivative order of twist to compute (>=0)
% %
% % ----------------------------
% % OUTPUT
% % ----------------------------
% % out.F{j}       : 4x4 configuration-dependent transform F_j (PoE part), with F_1 = C^0_1
% % out.A0{j}      : 4x4 constant transform A^0_j
% % out.C0{j}      : 4x4 world pose C^0_j = F_j * A^0_j
% % out.S{j}       : 6x(r+1) S_j^{(k)} for k=0..r (spatial screw in {0})
% % out.V{j}       : 6x(r+1) (V^0_j)^{(k)} for k=0..r
% %
% % ----------------------------
% % REQUIREMENTS / ASSUMPTIONS
% % ----------------------------
% % - Tree topology (each j>1 has exactly one parent p(j))
% % - 1-DoF joints for j>1 (universal joints should be modeled as consecutive 1-DoF joints)
% % - Spatial (right-invariant) representation; Ad/ad definitions below match [omega; v].
% 
% % ----------------------------
% % BASIC CHECKS
% % ----------------------------
% assert(iscell(c) && numel(c)==N, 'c must be 1xN cell.');
% assert(iscell(Ap_j) && numel(Ap_j)==N, 'Ap_j must be 1xN cell.');
% assert(iscell(Y) && numel(Y)==N, 'Y must be 1xN cell.');
% assert(all(size(C0_1)==[4 4]), 'C0_1 must be 4x4.');
% assert(all(size(A0_1)==[4 4]), 'A0_1 must be 4x4.');
% assert(all(size(V0_1_derivs)==[6 r+1]), 'V0_1_derivs must be 6x(r+1).');
% assert(size(q_derivs,1)==N && size(q_derivs,2)>=r+2, 'q_derivs must be Nx(r+2) or larger.');
% 
% % ----------------------------
% % ALLOCATE OUTPUT CONTAINERS
% % ----------------------------
% F  = cell(1,N);
% A0 = cell(1,N);
% C0 = cell(1,N);
% S  = cell(1,N);   % each S{j} is 6x(r+1)
% V  = cell(1,N);   % each V{j} is 6x(r+1)
% 
% for j = 1:N
%     S{j} = zeros(6, r+1);
%     V{j} = zeros(6, r+1);
% end
% 
% % ----------------------------
% % INITIALIZE BASE (j=1)
% % ----------------------------
% F{1}  = C0_1;     % as in algorithm: F_1 = C^0_1
% A0{1} = A0_1;     % constant
% C0{1} = C0_1;     % C^0_1 = F_1 * A^0_1 if A0_1=I; we keep as provided
% V{1}  = V0_1_derivs;
% 
% % (S{1} unused for floating base free joint here; keep zeros)
% 
% % Precompute binomial coefficients up to r (small and stable for modest r)
% % binom(n,k) = nchoosek(n,k)
% binom = cell(r+1,1);
% for n = 0:r
%     b = zeros(1,n+1);
%     for k = 0:n
%         b(k+1) = nchoosek(n,k);
%     end
%     binom{n+1} = b;
% end
% 
% % ----------------------------
% % RUN PREORDER DFS
% % ----------------------------
% Forward(1);
% 
% % Pack outputs
% out.F  = F;
% out.A0 = A0;
% out.C0 = C0;
% out.S  = S;
% out.V  = V;
% 
% % ==========================================================
% % NESTED FUNCTION: FORWARD RECURSION
% % ==========================================================
%     function Forward(j)
%         % Compute kinematics for node j (children handled after)
%         if j > 1
%             pj = p(j);
%             assert(pj>=1 && pj<=N, 'Invalid parent index for body %d.', j);
% 
%             % ----------------------------
%             % 0th order (pose) recursion
%             % ----------------------------
%             qj = q_derivs(j, 1);                 % q^{(0)}
%             F{j}  = F{pj} * expm_se3(Y{j} * qj); % F_j = F_p * exp([Y] q)
%             A0{j} = A0{pj} * Ap_j{j};            % A^0_j = A^0_p * A^{p}_j (constant)
%             C0{j} = F{j} * A0{j};                % C^0_j = F_j * A^0_j
% 
%             % ----------------------------
%             % S_j (spatial screw in {0})
%             % ----------------------------
%             % In your algorithm: S_j = Ad_{C0_j * (A0_j)^{-1}} Y_j = Ad_{F_j} Y_j
%             S{j}(:,1) = Ad_SE3(F{j}) * Y{j};     % S^{(0)}
% 
%             % ----------------------------
%             % V_j^{(0)} = V_p^{(0)} + S*qdot
%             % ----------------------------
%             qdot = q_derivs(j, 2);               % q^{(1)}
%             V{j}(:,1) = V{pj}(:,1) + S{j}(:,1) * qdot;
% 
%             % ----------------------------
%             % Higher-order derivatives up to r
%             % ----------------------------
%             % We compute in increasing order k = 1..r, so that:
%             % - S^{(k)} uses V^{(0..k-1)} and S^{(0..k-1)}
%             % - V^{(k)} uses S^{(0..k)} and q^{(1..k+1)}
%             for k = 1:r
%                 % ---- Compute S^{(k)} via:
%                 % S^{(k)} = sum_{r2=0}^{k-1} C(k-1,r2) * ad_{V^{(r2)}} * S^{(k-1-r2)}
%                 Sk = zeros(6,1);
%                 b = binom{k}; % corresponds to n = k-1, length k
%                 for r2 = 0:(k-1)
%                     coeff = b(r2+1); % C(k-1,r2)
%                     Sk = Sk + coeff * ( ad_twist(V{j}(:, r2+1)) * S{j}(:, (k-1-r2)+1) );
%                 end
%                 S{j}(:, k+1) = Sk;
% 
%                 % ---- Compute V^{(k)} via:
%                 % V^{(k)} = V_p^{(k)} + sum_{r2=0}^{k} C(k,r2) * S^{(r2)} * q^{(k-r2+1)}
%                 Vk = V{pj}(:, k+1); % parent derivative already computed (preorder)
%                 bk = binom{k+1};    % corresponds to n = k, length k+1
%                 for r2 = 0:k
%                     coeff = bk(r2+1); % C(k,r2)
%                     q_term_order = (k - r2 + 1); % derivative order of q
%                     q_term = q_derivs(j, q_term_order + 1); % +1 for 0-based -> column
%                     Vk = Vk + coeff * ( S{j}(:, r2+1) * q_term );
%                 end
%                 V{j}(:, k+1) = Vk;
%             end
%         end
% 
%         % Recurse on children (preorder DFS)
%         kids = c{j};
%         for ii = 1:numel(kids)
%             Forward(kids(ii));
%         end
%     end
% 
% end % function fkine_tree_higher_order
% 
% % ==========================================================
% % HELPER FUNCTIONS (LOCAL)
% % ==========================================================
% function T = expm_se3(xi_theta)
% % xi_theta is 6x1 twist multiplied by scalar (i.e., Y*q).
% % We form the 4x4 se(3) matrix and use expm for clarity.
% T = expm(hat_se3(xi_theta));
% end
% 
% 
% function Xi = hat_se3(xi)
% % Spatial twist to 4x4 se(3) matrix, xi = [omega; v]
% w = xi(1:3);
% v = xi(4:6);
% Xi = [skew3(w), v; 0 0 0 0];
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
% % 3x3 skew-symmetric matrix [x] such that [x]y = x x y
% S = [  0   -x(3)  x(2);
%       x(3)  0    -x(1);
%      -x(2) x(1)   0  ];
% end
% 
% % function out = GRNE_FBS_Forward( ...
% %     N, c, p, A0_1, Ap_j, C0_1, V0_1_derivs, q_derivs, Y, r)
% % %FKINE_TREE_HIGHER_ORDER Higher-order recursive forward kinematics (spatial) for floating-base trees.
% % %
% % % Spatial convention (as you stated): twist is [omega; v] with angular first.
% % % All twists/screws are expressed in the inertial frame {0} (spatial / right-invariant form).
% % %
% % % ----------------------------
% % % USER INPUTS (KEEP HERE)
% % % ----------------------------
% % % N              : number of bodies, indexed 1..N (body 1 is floating base)
% % % c              : 1xN cell, c{j} = array of children indices of body j
% % % p              : 1xN (or Nx1) parent index array; p(1) can be 0 or 1 (unused)
% % % A0_1           : 4x4 constant transform A^0_1 at home (often eye(4))
% % % Ap_j           : 1xN cell, Ap_j{j} = A^{p{j}}_j (4x4) for j>1; Ap_j{1} ignored
% % % C0_1           : 4x4 base pose C^0_1(t) in SE(3)
% % % V0_1_derivs    : 6x(r+1) base spatial twist derivatives:
% % %                  V0_1_derivs(:,k+1) = (V^0_1)^{(k)} for k=0..r
% % % q_derivs       : Nx(r+2) joint derivatives for each body index j:
% % %                  q_derivs(j,k+1) = q_j^{(k)} for k=0..(r+1)
% % %                  NOTE: To compute (V_j)^{(r)} you need q^{(r+1)} (hence r+2 columns).
% % %                  For body 1 (base), row is ignored (can be zeros).
% % % Y              : 1xN cell, Y{j} is 6x1 constant screw axis for joint j (j>1), at home
% % % r              : highest derivative order of twist to compute (>=0)
% % %
% % % ----------------------------
% % % OUTPUT
% % % ----------------------------
% % % out.F{j}       : 4x4 configuration-dependent transform F_j (PoE part), with F_1 = C^0_1
% % % out.A0{j}      : 4x4 constant transform A^0_j
% % % out.C0{j}      : 4x4 world pose C^0_j = F_j * A^0_j
% % % out.S{j}       : 6x(r+1) S_j^{(k)} for k=0..r (spatial screw in {0})
% % % out.V{j}       : 6x(r+1) (V^0_j)^{(k)} for k=0..r
% % %
% % % ----------------------------
% % % REQUIREMENTS / ASSUMPTIONS
% % % ----------------------------
% % % - Tree topology (each j>1 has exactly one parent p(j))
% % % - 1-DoF joints for j>1 (universal joints should be modeled as consecutive 1-DoF joints)
% % % - Spatial (right-invariant) representation; Ad/ad definitions below match [omega; v].
% % 
% % % ----------------------------
% % % BASIC CHECKS
% % % ----------------------------
% % assert(iscell(c) && numel(c)==N, 'c must be 1xN cell.');
% % assert(iscell(Ap_j) && numel(Ap_j)==N, 'Ap_j must be 1xN cell.');
% % assert(iscell(Y) && numel(Y)==N, 'Y must be 1xN cell.');
% % assert(all(size(C0_1)==[4 4]), 'C0_1 must be 4x4.');
% % assert(all(size(A0_1)==[4 4]), 'A0_1 must be 4x4.');
% % assert(all(size(V0_1_derivs)==[6 r+1]), 'V0_1_derivs must be 6x(r+1).');
% % assert(size(q_derivs,1)==N && size(q_derivs,2)>=r+2, 'q_derivs must be Nx(r+2) or larger.');
% % 
% % % ----------------------------
% % % ALLOCATE OUTPUT CONTAINERS
% % % ----------------------------
% % F  = cell(1,N);
% % A0 = cell(1,N);
% % C0 = cell(1,N);
% % S  = cell(1,N);   % each S{j} is 6x(r+1)
% % V  = cell(1,N);   % each V{j} is 6x(r+1)
% % 
% % for j = 1:N
% %     S{j} = zeros(6, r+1);
% %     V{j} = zeros(6, r+1);
% % end
% % 
% % % ----------------------------
% % % INITIALIZE BASE (j=1)
% % % ----------------------------
% % F{1}  = C0_1;     % as in algorithm: F_1 = C^0_1
% % A0{1} = A0_1;     % constant
% % C0{1} = C0_1;     % C^0_1 = F_1 * A^0_1 if A0_1=I; we keep as provided
% % V{1}  = V0_1_derivs;
% % 
% % % (S{1} unused for floating base free joint here; keep zeros)
% % 
% % % Precompute binomial coefficients up to r (small and stable for modest r)
% % % binom(n,k) = nchoosek(n,k)
% % binom = cell(r+1,1);
% % for n = 0:r
% %     b = zeros(1,n+1);
% %     for k = 0:n
% %         b(k+1) = nchoosek(n,k);
% %     end
% %     binom{n+1} = b;
% % end
% % 
% % % ----------------------------
% % % RUN PREORDER DFS
% % % ----------------------------
% % Forward(1);
% % 
% % % Pack outputs
% % out.F  = F;
% % out.A0 = A0;
% % out.C0 = C0;
% % out.S  = S;
% % out.V  = V;
% % 
% % % ==========================================================
% % % NESTED FUNCTION: FORWARD RECURSION
% % % ==========================================================
% %     function Forward(j)
% %         % Compute kinematics for node j (children handled after)
% %         if j > 1
% %             pj = p(j);
% %             assert(pj>=1 && pj<=N, 'Invalid parent index for body %d.', j);
% % 
% %             % ----------------------------
% %             % 0th order (pose) recursion
% %             % ----------------------------
% %             qj = q_derivs(j, 1);                 % q^{(0)}
% %             F{j}  = F{pj} * expm_se3(Y{j} * qj); % F_j = F_p * exp([Y] q)
% %             A0{j} = A0{pj} * Ap_j{j};            % A^0_j = A^0_p * A^{p}_j (constant)
% %             C0{j} = F{j} * A0{j};                % C^0_j = F_j * A^0_j
% % 
% %             % ----------------------------
% %             % S_j (spatial screw in {0})
% %             % ----------------------------
% %             % In your algorithm: S_j = Ad_{C0_j * (A0_j)^{-1}} Y_j = Ad_{F_j} Y_j
% %             S{j}(:,1) = Ad_SE3(F{j}) * Y{j};     % S^{(0)}
% % 
% %             % ----------------------------
% %             % V_j^{(0)} = V_p^{(0)} + S*qdot
% %             % ----------------------------
% %             qdot = q_derivs(j, 2);               % q^{(1)}
% %             V{j}(:,1) = V{pj}(:,1) + S{j}(:,1) * qdot;
% % 
% %             % ----------------------------
% %             % Higher-order derivatives up to r
% %             % ----------------------------
% %             % We compute in increasing order k = 1..r, so that:
% %             % - S^{(k)} uses V^{(0..k-1)} and S^{(0..k-1)}
% %             % - V^{(k)} uses S^{(0..k)} and q^{(1..k+1)}
% %             for k = 1:r
% %                 % ---- Compute S^{(k)} via:
% %                 % S^{(k)} = sum_{r2=0}^{k-1} C(k-1,r2) * ad_{V^{(r2)}} * S^{(k-1-r2)}
% %                 Sk = zeros(6,1);
% %                 b = binom{k}; % corresponds to n = k-1, length k
% %                 for r2 = 0:(k-1)
% %                     coeff = b(r2+1); % C(k-1,r2)
% %                     Sk = Sk + coeff * ( ad_twist(V{j}(:, r2+1)) * S{j}(:, (k-1-r2)+1) );
% %                 end
% %                 S{j}(:, k+1) = Sk;
% % 
% %                 % ---- Compute V^{(k)} via:
% %                 % V^{(k)} = V_p^{(k)} + sum_{r2=0}^{k} C(k,r2) * S^{(r2)} * q^{(k-r2+1)}
% %                 Vk = V{pj}(:, k+1); % parent derivative already computed (preorder)
% %                 bk = binom{k+1};    % corresponds to n = k, length k+1
% %                 for r2 = 0:k
% %                     coeff = bk(r2+1); % C(k,r2)
% %                     q_term_order = (k - r2 + 1); % derivative order of q
% %                     q_term = q_derivs(j, q_term_order + 1); % +1 for 0-based -> column
% %                     Vk = Vk + coeff * ( S{j}(:, r2+1) * q_term );
% %                 end
% %                 V{j}(:, k+1) = Vk;
% %             end
% %         end
% % 
% %         % Recurse on children (preorder DFS)
% %         kids = c{j};
% %         for ii = 1:numel(kids)
% %             Forward(kids(ii));
% %         end
% %     end
% % 
% % end % function fkine_tree_higher_order
% % 
% % % ==========================================================
% % % HELPER FUNCTIONS (LOCAL)
% % % ==========================================================
% % function T = expm_se3(xi_theta)
% % % xi_theta is 6x1 twist multiplied by scalar (i.e., Y*q).
% % % We form the 4x4 se(3) matrix and use expm for clarity.
% % T = expm(hat_se3(xi_theta));
% % end
% % 
% % 
% % function Xi = hat_se3(xi)
% % % Spatial twist to 4x4 se(3) matrix, xi = [omega; v]
% % w = xi(1:3);
% % v = xi(4:6);
% % Xi = [skew3(w), v; 0 0 0 0];
% % end
% % 
% % function A = Ad_SE3(T)
% % % Adjoint map Ad_T for SE(3), with twist [omega; v]
% % R = T(1:3,1:3);
% % p = T(1:3,4);
% % A = [R, zeros(3,3);
% %      skew3(p)*R, R];
% % end
% % 
% % function a = ad_twist(V)
% % % Lie algebra adjoint ad_V such that ad_V * W = V x W for twists [omega; v]
% % w = V(1:3);
% % v = V(4:6);
% % a = [skew3(w), zeros(3,3);
% %      skew3(v), skew3(w)];
% % end
% % 
% % function S = skew3(x)
% % % 3x3 skew-symmetric matrix [x] such that [x]y = x x y
% % S = [  0   -x(3)  x(2);
% %       x(3)  0    -x(1);
% %      -x(2) x(1)   0  ];
% % end