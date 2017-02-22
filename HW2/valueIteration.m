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
while 1
    A_s = zeros(size(R_sa));
    delta = 0;
    v_star = zeros(size(v_0));
    for s = 1 : size(v_0)
        Q_sa = zeros(size(R_sa, 2), 1);
        for a = 1 : size(R_sa, 2)
        Q_sa(a) = dot(P(s, :, a), v_0 * gamma + R_sa(s, a));
        end
        max_val = max(Q_sa);
        v_star(s) = max_val;
        delta = max(delta, abs(max_val - v_0(s)));
        eps = 0.001;
        max_index = abs(Q_sa-max_val) < eps;
        A_s(s, max_index) = 1;
    end
    if delta < theta
        break;
    end
    v_0 = v_star;
end
policy = zeros(size(A_s));
for i=1:size(policy) - 1
    policy(i,:) = A_s(i,:) / sum(A_s(i,:));
end
end