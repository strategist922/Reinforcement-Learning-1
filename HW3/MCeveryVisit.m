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

v_pi = initial_v_pi;
[statesFromEpisodes, actionsFromEpisodes, rewardsFromEpisodes] = getEpisodes(stateSpace, policy, num_episodes);
all_v_pi = zeros(size(stateSpace, 1), num_episodes);
count_states = zeros(size(stateSpace, 1), 1);
for i = 1 : num_episodes
states = cell2mat(statesFromEpisodes(i));
actions = cell2mat(actionsFromEpisodes(i));
rewards = cell2mat(rewardsFromEpisodes(i));
total_rewards = 0;
length_states = size(states, 1);
for j = length_states : -1: 1
    state_index = find( all( repmat(states(j, :), size(stateSpace, 1),1) == stateSpace, 2) );
    count_states(state_index) = count_states(state_index) + 1;
    if alpha == 0
        k = 1/count_states(state_index);
    else
        k = alpha;
    end
    v_pi(state_index) = v_pi(state_index) + k * (total_rewards - v_pi(state_index));
    if  j ~= 1
        total_rewards = total_rewards * gamma + rewards(j - 1);
    end
end
all_v_pi(:, i) = v_pi;
end

end