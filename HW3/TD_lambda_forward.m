function [v_pi, all_v_pi] = TD_lambda_forward(stateSpace, getEpisodes,policy, alpha, ...
    gamma, initial_v_pi, num_episodes,lambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function implements TD(lambda) algorithm from the forward view. It 
% estimates the state value function v(s) under a given policy.
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_states = size(stateSpace, 1);
v_pi = initial_v_pi;
all_v_pi = zeros(num_states, num_episodes);
% Initialization
[statesFromEpisodes, actionsFromEpisodes, rewardsFromEpisodes] = getEpisodes(stateSpace, policy, num_episodes);
for i = 1 : num_episodes
    % For every episode
    states = cell2mat(statesFromEpisodes(i));
    rewards = cell2mat(rewardsFromEpisodes(i));
    length_episode = size(states, 1);
    for j = 1 : length_episode - 1
        % For every non-terminate state
        state_index = find( all( repmat(states(j, :), num_states, 1) == stateSpace, 2) );
        num_remain_states = length_episode - j;
        % Calculate T
        G_n = zeros(num_remain_states, 1);
        for n = 1 : num_remain_states
            % Calculate every G_n for current state
            last_index = find( all( repmat(states(j + n, :), num_states, 1) == stateSpace, 2) );
            % Index of t + T
            x = 0 : n - 1;
            g = gamma .^ x;
            % Vectorization for efficiency
            G_n(n) = dot(g, rewards(j : j + n - 1)) + (gamma ^ n) * v_pi(last_index);
            % Save G_n
        end
        G_lambda = 0;
        % Initialization G_lambda
        if num_remain_states ~= 1
            % Ignore first term when T = 1
            x = 0 : num_remain_states - 2;
            l = lambda.^x;
            % Vectorization for efficiency
            G_lambda = G_lambda + (1 - lambda) * dot(l, G_n(1 : end - 1));
            % Calculate the first term in G_lambda
        end
        G_lambda = G_lambda + lambda .^ (num_remain_states - 1) * G_n(end);
        % Add the second term in G_lambda
        v_pi(state_index) = v_pi(state_index) + alpha * (G_lambda - v_pi(state_index));
        % Update v_pi according to G_lambda
    end
    all_v_pi(:, i) = v_pi;
    % Save intermediate results;
end
end