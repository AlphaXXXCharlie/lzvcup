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
% Soft Constraints Definitions & Penalties (Placeholder)
% We will define prioritized soft constraints here later.
% For now, it's the same as the original schedule.lp
%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

% Penalize consecutive home/away
consecHome(T,R) :- match(R,T,_), match(R+1,T,_), team(T), round(R), round(R+1).
consecAway(T,R) :- match(R,_,T), match(R+1,_,T), team(T), round(R), round(R+1).

#minimize { 1@1,T,R : consecHome(T,R) }.
#minimize { 1@1,T,R : consecAway(T,R) }.

%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
% Output
%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#show match/3.