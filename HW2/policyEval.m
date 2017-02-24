%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This funciton computes the value function under a given policy by using
% the fixed-point iteration method.
% Inputs: policy, transition probability matrix P, expected reward R_s^a
%         (also a matrix), discount factor gamma, threshold value theta,
%         and maximum number of iteration max_iter
% Output: valute function v_pi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function v_pi = policyEval(policy, P, R_sa, gamma, theta, max_iter)
% Constants
v_num = size(P, 1);
a_num = size(R_sa, 2);
% Initialization
v_pi = zeros(v_num, 1);
% For each iteration
for iter = 1 : max_iter
    v_pi_temp = zeros(size(v_pi));
    delta = 0;
    % For each state
    for s = 1 : v_num
        % For each action
        for a = 1 : a_num
            v_pi_temp(s) = v_pi_temp(s) + policy(s, a) * dot(P(s, :, a), v_pi * gamma + R_sa(s, a)); 
        end
        % Keep track of maximum change
        delta = max(delta, abs(v_pi(s) - v_pi_temp(s)));
    end
    % Record results
    v_pi = v_pi_temp;
    % Check if already stable
    if delta < theta
        break;
    end
end
end
%             t=3; 
%      State       v(s)
%     -----------------------
%          1      -2.4
%          2      -2.9
%          3      -3.0
%          4      -2.4
%          5      -2.9
%          6      -3.0
%          7      -2.9
%          8      -2.9
%          9      -3.0
%         10      -2.9
%         11      -2.4
%         12      -3.0
%         13      -2.9
%         14      -2.4
%         15       0.0
% 
% 
%             t=10; 
%      State       v(s)
%     -----------------------
%          1      -6.1
%          2      -8.4
%          3      -9.0
%          4      -6.1
%          5      -7.7
%          6      -8.4
%          7      -8.4
%          8      -8.4
%          9      -8.4
%         10      -7.7
%         11      -6.1
%         12      -9.0
%         13      -8.4
%         14      -6.1
%         15       0.0
% 
% 
%             t=1000; 
%      State       v(s)
%     -----------------------
%          1      -14.0
%          2      -20.0
%          3      -22.0
%          4      -14.0
%          5      -18.0
%          6      -20.0
%          7      -20.0
%          8      -20.0
%          9      -20.0
%         10      -18.0
%         11      -14.0
%         12      -22.0
%         13      -20.0
%         14      -14.0
%         15        0.0