function [v_pi, all_v_pi] = MCeveryVisit(stateSpace,getEpisodes,policy,...
    alpha, gamma, initial_v_pi, num_episodes)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function implements the Monte Carlo every visit algorithm. It
% estimates the state value function v(s) under a given policy.
% Inputs: stateSpace: matrix - 1st dimension is # of states, 2nd is # of
%           atributes for a given state. For example, every state in the
%           black jack example is a row vector with three elements.
%         getEpisodes: user-defined function that generates episodes
%           under the given policy
%         policy: matrix - 1st dimension is # of states, 2nd is # of
%           actions. policy(s,a) is pi(a-th action | s-th state)
%         alpha: learning rate. If alpha = 0, update v using v(s) = v(s) +
%           1/N(s)*(G - v(s)), otherwise use v(s) = v(s) + alpha*(G - v(s))
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
count_visits = zeros(num_states, 1);
% Initialization
[statesFromEpisodes, actionsFromEpisodes, rewardsFromEpisodes] = getEpisodes(stateSpace, policy, num_episodes);
% Get episodes
for i = 1 : num_episodes
    % For every episode
    states = cell2mat(statesFromEpisodes(i));
    rewards = cell2mat(rewardsFromEpisodes(i));
    % Transfer cells to matrics
    total_rewards = 0;
    % Keep track of total rewards
    length_episode = size(states, 1);
    for j = length_episode - 1 : -1: 1
        % Calculate backwardly for high efficiency
        total_rewards = total_rewards * gamma + rewards(j);
        state_index = find( all( repmat(states(j, :), num_states, 1) == stateSpace, 2) );
        % Find state index
        count_visits(state_index) = count_visits(state_index) + 1;
        % Add counter
        if alpha == 0
            learning_rate = 1 / count_visits(state_index);
        else
            learning_rate = alpha;
        end
        % Choose learning rate according to alpha
        v_pi(state_index) = v_pi(state_index) + learning_rate * (total_rewards - v_pi(state_index));
        % Update v_pi
    end
    all_v_pi(:, i) = v_pi;
    % Save intermediate results
end
end