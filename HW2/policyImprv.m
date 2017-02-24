%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This funciton improves a given policy.
% Inputs: transition probability matrix P, expected reward R_s^a, discount
%         factor gamma, allowed action(s) and the value function v_pi for
%         each state under current policy      
% Outputs: improved policy and the allowed action(s) for each state;
%          boolean variable policy_stable
% Note: A_s is a boolean (0 & 1) matrix. Its (s,a) entry is 1 if it's 
%       possible to take the a-th action at the s-th state UNDER the 
%       current policy.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [policy_new,A_s_new,policy_stable] = policyImprv(P,R_sa,gamma,...
            A_s,v_pi)
% Constant
v_num = size(P, 1);
a_num = size(R_sa, 2);
% Initialiaztion
policy_stable = 1;
policy_new = zeros(v_num, a_num);
A_s_new = zeros(size(A_s));
% For each state
for s = 1 : v_num - 1
    % Record each action value
    Q_sa = zeros(a_num, 1);
    % For each action
    for a = 1 : a_num
    Q_sa(a) = dot(P(s, :, a), v_pi * gamma + R_sa(s, a));
    end
    % Extract maximum value and corresponding indices
    max_val = max(Q_sa);
    max_index = Q_sa == max_val;
    % Set actions
    A_s_new(s, max_index) = 1;
    % Check if stable
    if ~isequal(A_s_new(s,:), A_s(s,:))
    policy_stable = 0;
    end
end
% Calculate policy values
for i=1:size(policy_new) - 1
    policy_new(i,:) = A_s_new(i,:) / sum(A_s_new(i,:));
end
end

%                 Optimal policy
%    State     Left       Up    Right     Down
% -----------------------------------------------
%        1     1.00     0.00     0.00     0.00
%        2     1.00     0.00     0.00     0.00
%        3     0.50     0.00     0.00     0.50
%        4     0.00     1.00     0.00     0.00
%        5     0.50     0.50     0.00     0.00
%        6     0.25     0.25     0.25     0.25
%        7     0.00     0.00     0.00     1.00
%        8     0.00     1.00     0.00     0.00
%        9     0.25     0.25     0.25     0.25
%       10     0.00     0.00     0.50     0.50
%       11     0.00     0.00     0.00     1.00
%       12     0.00     0.50     0.50     0.00
%       13     0.00     0.00     1.00     0.00
%       14     0.00     0.00     1.00     0.00
% 
% 
%     Optimal value function
%      State       v(s)
%     -----------------------
%          1       -1.0
%          2       -2.0
%          3       -3.0
%          4       -1.0
%          5       -2.0
%          6       -3.0
%          7       -2.0
%          8       -2.0
%          9       -3.0
%         10       -2.0
%         11       -1.0
%         12       -3.0
%         13       -2.0
%         14       -1.0
%         15        0.0