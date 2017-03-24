function [v_pi, all_v_pi] = TD0(stateSpace, getEpisodes,policy, alpha, ...
    gamma, initial_v_pi, num_episodes)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function implements TD(0) algorithm. It estimates the state value
% function v(s) under a given policy.
% Inputs: stateSpace: matrix - 1st dimension is # of states, 2nd is # of
%           atributes for a given state. For example, every state in the
%           black jack example is a row vector with three elements.
%         getEpisodes: user-defined function that generates episodes
%           under the given policy
%         policy: matrix - 1st dimension is # of states, 2nd is # of
%           actions. policy(s,a) is pi(a-th action | s-th state)
%         alpha: learning rate
%         gamma: discount factor
%         initial_v_pi: vector - initial estimates for v_pi
%         num_episodes: number of episodes for the evaluation
% Outputs: v_pi: vector - final estimates for v_pi
%          all_v_pi: matrix of size #states x #episodes. It stores all
%                    estimates for v_pi after every episode
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_states = size(stateSpace, 1);
v_pi = initial_v_pi;
all_v_pi = zeros(num_states, num_episodes);
% Initialization
[statesFromEpisodes, actionsFromEpisodes, rewardsFromEpisodes] = getEpisodes(stateSpace, policy, num_episodes);
% Get episodes
for i = 1 : num_episodes
    % For every episode
    states = cell2mat(statesFromEpisodes(i));
    rewards = cell2mat(rewardsFromEpisodes(i));
    % Convert cells to matrics
    length_episode = size(states, 1);
    state_index = find( all( repmat(states(1, :), num_states, 1) == stateSpace, 2) );
    % Find index of the first state
    for j = 1 : length_episode - 1
        % For every state in the episode except for the terminate state
        next_state_index = find( all( repmat(states(j + 1, :), num_states, 1) == stateSpace, 2) );
        % Find index of the next state
        v_pi(state_index) = v_pi(state_index) + alpha * (rewards(j) + gamma * v_pi(next_state_index) - v_pi(state_index));
        % Update v_pi
        state_index = next_state_index;
        % Update state index
    end
    all_v_pi(:, i) = v_pi;
    % Save intermediate results
end
end