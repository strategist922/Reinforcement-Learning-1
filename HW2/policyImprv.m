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
policy_stable = 1;
policy_new = zeros(size(R_sa));
A_s_new = zeros(size(A_s));
for s = 1 : size(A_s, 1)
    Q_as = zeros(size(R_sa, 2), 1);
    for a = 1 : size(R_sa, 2)
    Q_as(a) = dot(P(s, :, a), v_pi * gamma + R_sa(s, a));
    end
    max_val = max(Q_as);
    eps = 0.001;
    max_index = abs(Q_as-max_val) < eps;
    A_s_new(s, max_index) = 1;
    if ~isequal(A_s_new(s,:), A_s(s,:))
    policy_stable = 0;
    end
end
for i=1:size(policy_new) - 1
    policy_new(i,:) = A_s_new(i,:) / sum(A_s_new(i,:));
end
end