%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This funciton computes the value function under a given policy by using
% the fixed-point iteration method.
% Inputs: policy, transition probability matrix P, expected reward R_s^a
%         (also a matrix), discount factor gamma, threshold value theta,
%         and maximum number of iteration max_iter
% Output: valute function v_pi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function v_pi = policyEval(policy, P, R_sa, gamma, theta, max_iter)
v_pi = zeros(size(P, 1), 1);
for iter = 1 : max_iter
    v_pi_temp = zeros(size(v_pi));
    delta = 0;
    for s = 1 : size(v_pi)
        for a = 1 : size(R_sa, 2)
            v_pi_temp(s) = v_pi_temp(s) + policy(s, a) * dot(P(s, :, a), v_pi * gamma + R_sa(s, a)); 
        end
        delta = max(delta, v_pi(s) - v_pi_temp(s));
    end
    v_pi = v_pi_temp;
    if delta < theta
        break;
    end
end
end