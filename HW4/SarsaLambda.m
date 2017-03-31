function [Q, policy, steps] = SarsaLambda(stateSpace, initialStates, ...
    terminalStates, takeAction, initialPolicy, numIterations, gamma, alpha,lambda,epsilon)
% This function implements Sarsa(lambda) algorithm.
% Inputs: stateSapce - matrix with each row representing a state
%         initialStates - matrix containing all possible initial states
%         terminalStates - matrix containing all terminal states
%         takeAction - function that gives S' and reward for the current
%                      state-action pair
%         numIterations - number of total iterations (episodes)
%         gamma - discout factor
%         alpha - learning rate (constant for simplicity)
%         lambda - parameter in Q(lambda)
%         epsilon - parameter in epislon-greedy policy (constant for
%                   simplicity)
% Outputs: Q - matrix containing q-value for each (s,a)
%          policy - matrix containing the probabilities for taking each
%                   action at each state
%          steps - vector, stores number of steps in each episode

policy = initialPolicy;
Q = zeros(size(initialPolicy));
steps = zeros(numIterations, 1);
actionsNum = size(initialPolicy, 2);
statesNum = size(stateSpace, 1);
prob_max = epsilon / actionsNum + 1 - epsilon;
prob_other = epsilon / actionsNum;
for n = 1 : numIterations
    step = 1;
    E = zeros(size(initialPolicy));
    S = initialStates(randsample(size(initialStates, 1),1), :);
    S_index = find( all( repmat(S, statesNum, 1) == stateSpace, 2) );
    prob = policy(S_index, :);
    A = sum(rand() >= cumsum([0, prob]));
    while ~isequal(S, terminalStates)
        %GENERALIZA!!!!!!!!
        [next_S, R] = takeAction(S, A);
        S_index = find( all( repmat(S, statesNum, 1) == stateSpace, 2) );
        next_S_index = find( all( repmat(next_S, statesNum, 1) == stateSpace, 2) );
        next_Q = Q(next_S_index, :);
        max_index = find (next_Q == max(next_Q), 1, 'first');
        policy(next_S_index, :) = ones(1, actionsNum).*prob_other;
        policy(next_S_index, max_index) = prob_max;
        prob = policy(next_S_index, :);
        next_A = sum(rand() >= cumsum([0, prob]));
        delta = R + gamma*Q(next_S_index, next_A) - Q(S_index,A);
        E(S_index, A) = E(S_index, A) + 1;
        for state = 1 : statesNum
            for action = 1 : actionsNum
                Q(state, action) = Q(state, action) + alpha * delta * E(state, action);
            end
        end
        E = E * lambda * gamma;
        S = next_S;
        A = next_A;
        step = step + 1;
    end
    steps(n) = step;
end
end


