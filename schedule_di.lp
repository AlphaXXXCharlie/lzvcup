%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% schedule_di.lp
% Enhanced Double Round‐Robin Scheduling for LZV Cup (DI Version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
% Parameters are expected from an instance .lp file (e.g., #const n = 4.)
%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
team(1..n).
% total_rounds = 2*(n-1). % You can keep this as a comment for clarity
round(1..2*(n-1)).      % Use the formula directly

%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
% Hard Constraints
%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

% Each unordered pair plays once with A@home and once with B@home
1 { match(R,A,B) : round(R) } 1 :- team(A), team(B), A < B.
1 { match(R,B,A) : round(R) } 1 :- team(A), team(B), A < B.

% Each team plays exactly one match per round
:- team(T), round(R),
   PlayedHome = #count{ Opponent : match(R,T,Opponent) },
   PlayedAway = #count{ Opponent : match(R,Opponent,T) },
   PlayedHome + PlayedAway != 1.

%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
% Soft Constraints Definitions & Penalties
% Higher priority values are optimized first.
% Higher penalty costs within the same priority are more impactful.
%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

% --- S1: Avoid three or more consecutive home/away matches (High Priority) ---
% FOr my own Reference: This is often a very undesirable feature.
consecutive_home_3(T,R) :- team(T), round(R), round(R+1), round(R+2),
                           match(R,T,_), match(R+1,T,_), match(R+2,T,_).
consecutive_away_3(T,R) :- team(T), round(R), round(R+1), round(R+2),
                           match(R,_,T), match(R+1,_,T), match(R+2,_,T).

% High penalty cost (e.g., 10) at a high priority level (e.g., 3)
#minimize { 10@3,T,R : consecutive_home_3(T,R) }.
#minimize { 10@3@3,T,R : consecutive_away_3(T,R) }. % Corrected typo: Removed extra @3


% --- S2: Minimize (any) two consecutive home/away matches (Medium Priority) ---
% This is less severe than three, so lower penalty or lower priority.
consecutive_home_2(T,R) :- team(T), round(R), round(R+1),
                           match(R,T,_), match(R+1,T,_).
consecutive_away_2(T,R) :- team(T), round(R), round(R+1),
                           match(R,_,T), match(R+1,_,T).

% Lower penalty cost (e.g., 1) at a medium priority level (e.g., 2)
#minimize { 1@2,T,R : consecutive_home_2(T,R) }.
#minimize { 1@2,T,R : consecutive_away_2(T,R) }.


% --- S3: Balance overall home/away count (Lower Priority) ---
% We will define this later.


%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
% Output
%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#show match/3.