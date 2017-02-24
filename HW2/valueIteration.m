%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function implements the value iteration algorithm.
% Inputs: transition probability matrix P, expected reward R_s^a, discount
%         factor gamma, threshold value theta, and initial value function
%         v_0
% Outputs: optimal policy, optimal action(s) A_s for each state, optimal
%         valute function v_star
% Note: A_s is a boolean (0 & 1) matrix. Its (s,a) entry is 1 if it's 
%       possible to take the a-th action at the s-th state UNDER the 
%       current policy.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [policy,A_s,v_star] = valueIteration(P, R_sa, gamma, theta,v_0)
% Constants
v_num = size(P, 1);
a_num = size(R_sa, 2);
% Run iteratively till it's stable
while 1
    % Initialiaztion
    A_s = zeros(v_num, a_num);
    delta = 0;
    v_star = zeros(v_num, 1);
    % For each state
    for s = 1 : v_num
        Q_sa = zeros(a_num, 1);
        % For each action, record its action value.
        for a = 1 : a_num
        Q_sa(a) = dot(P(s, :, a), v_0 * gamma + R_sa(s, a));
        end
        % Set V(s) to the maximum action value
        v_star(s) = max(Q_sa);
        % Keep track of maximum change
        delta = max(delta, abs(v_star(s) - v_0(s)));
    end
    % Check if stable
    if delta < theta
        break;
    end
    % Record results from this iteration
    v_0 = v_star;
end
% Calculate A_s 
for s = 1 : v_num
    Q_sa = zeros(a_num, 1);
    for a = 1 : a_num
        Q_sa(a) = dot(P(s, :, a), v_0 * gamma + R_sa(s, a));
    end
    % Get the index of maximum action value
    max_index = Q_sa == v_star(s);
    % Set actions
    A_s(s, max_index) = 1;
end
% Calculate policy value
policy = zeros(size(A_s));
for i=1:size(policy) - 1
    policy(i,:) = A_s(i,:) / sum(A_s(i,:));
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