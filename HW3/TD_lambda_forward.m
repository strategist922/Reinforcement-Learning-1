function [v_pi, all_v_pi] = TD_lambda_forward(stateSpace, getEpisodes,policy, alpha, ...
                            gamma, initial_v_pi, num_episodes,lambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function implements TD(lambda) algorithm. It estimates the state value 
% function v(s) under a given policy.
% Inputs: stateSpace: matrix - 1st dimension is # of states, 2nd is # of
%           atributes for a given state. For example, every state in the
%           black jack example is a row vector with three elements.
%         getEpisodes: user-defined function that generates episodes 
%           under the given policy
%         policy: matrix - 1st dimension is # of states, 2nd is # of
%           actions. policy(s,a) is pi(a-th action | s-th state)
%         alpha: learning rate. 
%         gamma: discount factor
%         initial_v_pi: vector - initial estimates for v_pi
%         num_episodes: number of episodes for the evaluation 
%         lambda: parameter in TD(lambda)
% Outputs: v_pi: vector - final estimates for v_pi
%          all_v_pi: matrix of size #states x #episodes. It stores all 
%                    estimates for v_pi after every episode 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


v_pi = initial_v_pi;
[statesFromEpisodes, actionsFromEpisodes, rewardsFromEpisodes] = getEpisodes(stateSpace, policy, num_episodes);
all_v_pi = zeros(size(stateSpace, 1), num_episodes);
for i = 1 : num_episodes
states = cell2mat(statesFromEpisodes(i));
rewards = cell2mat(rewardsFromEpisodes(i));
length_states = size(states, 1);
for j = 1 : length_states - 1
    state_index = find( all( repmat(states(j, :), size(stateSpace, 1),1) == stateSpace, 2) );
    number_remain = length_states - j;
    G_n = zeros(number_remain, 1);
    for n = 1 : number_remain
        last_index = find( all( repmat(states(j + n, :), size(stateSpace, 1),1) == stateSpace, 2) );
        x = 0 : n - 1;
        g = gamma.^x;
        G_n(n) = dot(g , rewards(j:j + n - 1)) + (gamma^n)*v_pi(last_index);
    end
    G_lambda = 0;
    if number_remain ~= 1
        x = 0 : number_remain - 2;
        l = lambda.^x;
        G_lambda = G_lambda + (1 - lambda)*dot(l , G_n(1:end - 1));
    end
    G_lambda = G_lambda + lambda.^(number_remain - 1)*G_n(end);
    v_pi(state_index) = v_pi(state_index) + alpha * (G_lambda - v_pi(state_index));
end
all_v_pi(:, i) = v_pi;
end

end