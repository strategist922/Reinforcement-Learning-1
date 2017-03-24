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
all_v_pi = zeros(size(stateSpace, 1), num_episodes);
num_states = size(stateSpace, 1);
% Initialization
[statesFromEpisodes, actionsFromEpisodes, rewardsFromEpisodes] = getEpisodes(stateSpace, policy, num_episodes);
% Get episodes
for i = 1 : num_episodes
    % For every episode
    eligibility_trace = zeros(num_states, 1);
    % Initialize eligibility trace
    states = cell2mat(statesFromEpisodes(i));
    rewards = cell2mat(rewardsFromEpisodes(i));
    length_episode = size(states, 1);
    state_index = find( all( repmat(states(1, :), num_states, 1) == stateSpace, 2) );
    % Find the index of the first state
    for j = 1 : length_episode - 1
        % For every non-terminate states
        eligibility_trace(state_index) = eligibility_trace(state_index) + 1;
        % Update eligibility trace
        next_state_index = find( all( repmat(states(j + 1, :), num_states, 1) == stateSpace, 2) );
        % Find the index of the next state
        error = rewards(j) + gamma * v_pi(next_state_index) - v_pi(state_index);
        % Calculate TD error
        for state = 1 : num_states
            v_pi(state) = v_pi(state) + alpha * error * eligibility_trace(state);
            % Update v_pi
        end
        state_index = next_state_index;
        % Update state index
        eligibility_trace = eligibility_trace * lambda * gamma;
        % Update eligibility trace
    end
    all_v_pi(:, i) = v_pi;
    % Save intermediate results
end
end