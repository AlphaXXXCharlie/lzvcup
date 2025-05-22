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
% Soft Constraints
%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

% Penalize consecutive home/away
consecHome(T,R) :- match(R,T,_), match(R+1,T,_), team(T), round(R), round(R+1).
consecAway(T,R) :- match(R,_,T), match(R+1,_,T), team(T), round(R), round(R+1).

#minimize { 1@1,T,R : consecHome(T,R) }.
#minimize { 1@1,T,R : consecAway(T,R) }.

% Balance home/away count (This is a simple version. More advanced balancing might be needed)
% homeCount(T,H) :- H = #count{ R : match(R,T,_) }, team(T).
% awayCount(T,A) :- A = #count{ R : match(R,_,T) }, team(T).
% imbalance(T,D) :- homeCount(T,H), awayCount(T,A), D = |H - A|, team(T).
% #minimize { D@2,T : imbalance(T,D) }.
% Note: The imbalance constraint above can sometimes conflict heavily with
% consecutive home/away for small N, or make finding solutions harder.
% You might want to adjust its priority or formulation if issues arise.
% For now, let's focus on the primary match generation.

%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
% Output
%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#show match/3.