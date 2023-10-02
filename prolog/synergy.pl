:- module(synergy, [heroes_synergy/2, heroes_synergy/3]).
:- use_module(abilities).
:- use_module(heroes).

% Rule that returns true if there is synergy between hero1 and hero2
% To simplify lets agree that there is synergy between two heroes if
% - one of them has ability with ability_type 'stun_single' and another one has ability with type 'dmg_single'
% - one of them has ability with ability_type 'stun_aoe' and another one has ability with type 'dmg_aoe'
heroes_synergy(HERO1, HERO2) :-
    \+(HERO1 = HERO2),
    hero_ability_type(HERO1, TYPE1),
    hero_ability_type(HERO2, TYPE2),
    TYPE1 = dmg_single,
    TYPE2 = stun_single.
heroes_synergy(HERO1, HERO2) :-
    \+(HERO1 = HERO2),
    hero_ability_type(HERO1, TYPE1),
    hero_ability_type(HERO2, TYPE2),
    TYPE1 = stun_single,
    TYPE2 = dmg_single.
heroes_synergy(HERO1, HERO2) :-
    \+(HERO1 = HERO2),
    hero_ability_type(HERO1, TYPE1),
    hero_ability_type(HERO2, TYPE2),
    TYPE1 = dmg_aoe,
    TYPE2 = stun_aoe.
heroes_synergy(HERO1, HERO2) :-
    \+(HERO1 = HERO2),
    hero_ability_type(HERO1, TYPE1),
    hero_ability_type(HERO2, TYPE2),
    TYPE1 = stun_aoe,
    TYPE2 = dmg_aoe.
heroes_synergy(HERO1, HERO2, HERO3) :-
    heroes_synergy(HERO1, HERO2),
    heroes_synergy(HERO2, HERO3),
    heroes_synergy(HERO1, HERO3).
