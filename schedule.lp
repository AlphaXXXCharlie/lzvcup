%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% schedule.lp
% Double Round‐Robin Scheduling for LZV Cup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
% Parameters are expected from an instance .lp file (e.g., #const n = 4.)
%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
team(1..n).
round(1..2*(n-1)).

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
% Output (basic)
%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#show match/3.