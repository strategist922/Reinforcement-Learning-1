%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function implements the policy iteration algorithm.
% Inputs: transition probability matrix P, expected reward R_s^a, discount
%         factor gamma, threshold value theta, maximum number of 
%         of iteration (in policy evalution) max_iter, initial policy and 
%         action(s) for each state under the initial policy
% Outputs: optimal policy, optimal action(s) A_s for each state, optimal
%         valute function v_pi, and total number of iterations until 
%         achieving the optimal policy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [policy, A_s, v_pi,num_iter] = policyIteration(P, R_sa, gamma, theta,...
             max_iter,init_policy,init_A_s)
    % Initialization
    policy_stable = 0;
    A_s = init_A_s;
    policy = init_policy; 
    
    num_iter=0;
    
    % main algorithm
    while ~policy_stable
        num_iter = num_iter+1;
        % policy evaluation
        v_pi = policyEval(policy, P, R_sa, gamma, theta, max_iter);
        % policy improvement
        [policy,A_s,policy_stable] = policyImprv(P,R_sa,gamma,A_s,v_pi);
    end
end