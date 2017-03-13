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

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your code 
%%%%%%%%%%%%%%%%%%%%%%%%%%%

end