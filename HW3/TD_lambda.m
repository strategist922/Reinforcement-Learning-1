function [v_pi, all_v_pi] = TD_lambda(stateSpace, getEpisodes,policy, alpha, ...
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
trace = zeros(size(stateSpace, 1), 1);
states = cell2mat(statesFromEpisodes(i));
rewards = cell2mat(rewardsFromEpisodes(i));
length_states = size(states, 1);
state_index = find( all( repmat(states(1, :), size(stateSpace, 1),1) == stateSpace, 2) );
for j = 1 : length_states
    trace(state_index) = trace(state_index) + 1;
    if j ~= length_states
        next_state_index = find( all( repmat(states(j + 1, :), size(stateSpace, 1),1) == stateSpace, 2) );
        reward = rewards(j);
    else
        next_state_index = state_index;
        reward = 0;
    end
    e = reward + gamma * v_pi(next_state_index) - v_pi(state_index);
    for state = 1 : size(stateSpace, 1)
        v_pi(state) = v_pi(state) + alpha * e * trace(state);
    end
    state_index = next_state_index;
    trace = trace * lambda * gamma;
end
all_v_pi(:, i) = v_pi;
end

end