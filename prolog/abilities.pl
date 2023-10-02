:- module(abilities, [ability/2, ability_type/2, hero_ability_type/2, hero_with_stun/1, hero_with_dmg/1, universal_hero/1]).
:- use_module(heroes).


% ability(ability_name, hero_name)
ability("BERSERKER'S CALL", "AXE").
ability("FIEND'S GRIP", "BANE").
ability("FLAMING LASSO", "BATRIDER").
ability("PRIMAL ROAR", "BEASTMASTER").
ability("FREEZING FIELD", "CRYSTAL MAIDEN").
ability("VACUUM", "DARK SEER").
ability("BEDLAM", "DARK WILLOW").
ability("TERRORIZE", "DARK WILLOW").
ability("MAGNETIZE", "EARTH SPIRIT").
ability("ECHO SLAM", "EARTHSHAKER").
ability("ECHO STOMP", "ELDER TITAN").
ability("EARTH SPLITTER", "ELDER TITAN").
ability("IMPETUS", "ENCHANTRESS").
ability("BLACK HOLE", "ENIGMA").
ability("CHRONOSPHERE", "FACELESS VOID").
ability("CALL DOWN", "GYROCOPTER").
ability("SHARPSHOOTER", "HOODWINK").
ability("LIFE BREAK", "HUSKAR").
ability("LIFE BREAK AGHANIM'S SCEPTER", "HUSKAR").
ability("SUN STRIKE", "INVOKER").
ability("SUN STRIKE AGHANIM'S SCEPTER", "INVOKER").
ability("ICE PATH", "JAKIRO").
ability("MACROPYRE", "JAKIRO").
ability("OMNISLASH", "JUGGERNAUT").
ability("DUEL", "LEGION COMMANDER").
ability("LAGUNA BLADE", "LINA").
ability("FINGER OF DEATH", "LION").
ability("REVERSE POLARITY", "MAGNUS").
ability("WUKONG'S COMMAND", "MONKEY KING").

% ability_type(ability_name, ability_type), where ability type is one of [dmg_single, dmg_aoe, stun_single, stun_aoe]
% dmg_single - huge damage to solo target
% dmg_aoe - huge damage to big area
% stun_single - great control to one target
% stun_single - great control to big area
ability_type("BERSERKER'S CALL", stun_aoe).
ability_type("FIEND'S GRIP", stun_single).
ability_type("FLAMING LASSO", stun_single).
ability_type("PRIMAL ROAR", stun_single).
ability_type("FREEZING FIELD", dmg_aoe).
ability_type("VACUUM", stun_aoe).
ability_type("BEDLAM", dmg_single).
ability_type("TERRORIZE", stun_aoe).
ability_type("MAGNETIZE", dmg_aoe).
ability_type("ECHO SLAM", stun_aoe).
ability_type("ECHO SLAM", dmg_aoe).
ability_type("ECHO STOMP", stun_aoe).
ability_type("EARTH SPLITTER", dmg_aoe).
ability_type("IMPETUS", dmg_single).
ability_type("BLACK HOLE", stun_aoe).
ability_type("BLACK HOLE", dmg_aoe).
ability_type("CHRONOSPHERE", stun_aoe).
ability_type("CALL DOWN", dmg_aoe).
ability_type("SHARPSHOOTER", dmg_single).
ability_type("LIFE BREAK", dmg_single).
ability_type("LIFE BREAK AGHANIM'S SCEPTER", stun_single).
ability_type("SUN STRIKE", dmg_single).
ability_type("SUN STRIKE AGHANIM'S SCEPTER", dmg_aoe).
ability_type("ICE PATH", stun_aoe).
ability_type("MACROPYRE", dmg_aoe).
ability_type("OMNISLASH", dmg_aoe).
ability_type("OMNISLASH", dmg_single).
ability_type("DUEL", stun_single).
ability_type("LAGUNA BLADE", dmg_single).
ability_type("FINGER OF DEATH", dmg_single).
ability_type("REVERSE POLARITY", stun_aoe).
ability_type("WUKONG'S COMMAND", dmg_aoe).

% Get hero ability types
hero_ability_type(HERO, TYPE) :- hero(HERO), ability(ABILITY, HERO), ability_type(ABILITY, TYPE).
hero_with_stun(HERO) :- hero_ability_type(HERO, TYPE), (TYPE = stun_single; TYPE = stun_aoe).
hero_with_dmg(HERO) :- hero_ability_type(HERO, TYPE), (TYPE = dmg_single; TYPE = dmg_aoe).
% Hero with both stun skill and dmg skill
universal_hero(HERO) :- hero_with_dmg(HERO), hero_with_stun(HERO).
