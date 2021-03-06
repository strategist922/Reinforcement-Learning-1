%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
% Problem Setup:
% 15 states in total; the last (15th) state is the terminal state(s)
% 4 possible actions: 1:left, 2:up, 3:right, 4:down
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear 

% Initializazion (random policy)
v_0 = zeros(15,1);  
% Discount factor 
gamma = 1;
% Expected reward
R_sa = -ones(15,4);
R_sa(15,:) = 0;
% Transition Probabilities P_ss'^a
P = zeros(15,15,4); 
P(15,15,:)=1;
    % for action = 1 (left)
    P(1,15,1)=1; P(2,1,1)=1; P(3,2,1)=1; P(4,4,1)=1; P(5,4,1)=1;
    P(6,5,1)=1; P(7,6,1)=1; P(8,8,1)=1; P(9,8,1)=1; P(10,9,1)=1;
    P(11,10,1)=1; P(12,12,1)=1; P(13,12,1)=1; P(14,13,1)=1;
    % for action = 2 (up)
    P(1,1,2)=1; P(2,2,2)=1; P(3,3,2)=1; P(4,15,2)=1;
    for i=5:14
        P(i,i-4,2)=1;
    end
    % for action = 3 (right)
    P(1,2,3)=1; P(2,3,3)=1; P(3,3,3)=1; P(4,5,3)=1; P(5,6,3)=1;
    P(6,7,3)=1; P(7,7,3)=1; P(8,9,3)=1; P(9,10,3)=1; P(10,11,3)=1;
    P(11,11,3)=1; P(12,13,3)=1; P(13,14,3)=1; P(14,15,3)=1;
    % for action = 4 (down)
    for i=1:10
        P(i,i+4,4)=1;
    end
    P(11,15,4)=1; P(12,12,4)=1; P(13,13,4)=1; P(14,14,4)=1;
   
theta = 1e-6;
[policy,A_s,v_star] = valueIteration(P, R_sa, gamma, theta,v_0);

fprintf('                Optimal policy\n')
fprintf('%8s %8s %8s %8s %8s\n', 'State', 'Left','Up','Right','Down')
fprintf('-----------------------------------------------\n')
fprintf('%8d %8.2f %8.2f %8.2f %8.2f\n', [1:14; policy(1:14,:)'])
fprintf('\n\n')
fprintf('    Optimal value function\n')
fprintf('%10s %10s\n', 'State', 'v(s)')
fprintf('    -----------------------\n')
fprintf('%10d %10.1f\n', [1:15; v_star'])