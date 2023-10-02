:- module(abilities, [hero_ability/2, ability_type/2, hero_ability_type/2, hero_with_stun/1, hero_with_dmg/1, universal_hero/1]).
:- use_module(heroes).

% ability(ability_name)
ability("BERSERKER'S CALL").
ability("FIEND'S GRIP").
ability("FLAMING LASSO").
ability("PRIMAL ROAR").
ability("FREEZING FIELD").
ability("VACUUM").
ability("BEDLAM").
ability("TERRORIZE").
ability("MAGNETIZE").
ability("ECHO SLAM").
ability("ECHO STOMP").
ability("EARTH SPLITTER").
ability("IMPETUS").
ability("BLACK HOLE").
ability("CHRONOSPHERE").
ability("CALL DOWN").
ability("SHARPSHOOTER").
ability("LIFE BREAK").
ability("LIFE BREAK AGHANIM'S SCEPTER").
ability("SUN STRIKE").
ability("SUN STRIKE AGHANIM'S SCEPTER").
ability("ICE PATH").
ability("MACROPYRE").
ability("OMNISLASH").
ability("DUEL").
ability("LAGUNA BLADE").
ability("FINGER OF DEATH").
ability("REVERSE POLARITY").
ability("WUKONG'S COMMAND").

% hero_ability(ability_name, hero_name)
hero_ability(ability("BERSERKER'S CALL"), hero("AXE")).
hero_ability(ability("FIEND'S GRIP"), hero("BANE")).
hero_ability(ability("FLAMING LASSO"), hero("BATRIDER")).
hero_ability(ability("PRIMAL ROAR"), hero("BEASTMASTER")).
hero_ability(ability("FREEZING FIELD"), hero("CRYSTAL MAIDEN")).
hero_ability(ability("VACUUM"), hero("DARK SEER")).
hero_ability(ability("BEDLAM"), hero("DARK WILLOW")).
hero_ability(ability("TERRORIZE"), hero("DARK WILLOW")).
hero_ability(ability("MAGNETIZE"), hero("EARTH SPIRIT")).
hero_ability(ability("ECHO SLAM"), hero("EARTHSHAKER")).
hero_ability(ability("ECHO STOMP"), hero("ELDER TITAN")).
hero_ability(ability("EARTH SPLITTER"), hero("ELDER TITAN")).
hero_ability(ability("IMPETUS"), hero("ENCHANTRESS")).
hero_ability(ability("BLACK HOLE"), hero("ENIGMA")).
hero_ability(ability("CHRONOSPHERE"), hero("FACELESS VOID")).
hero_ability(ability("CALL DOWN"), hero("GYROCOPTER")).
hero_ability(ability("SHARPSHOOTER"), hero("HOODWINK")).
hero_ability(ability("LIFE BREAK"), hero("HUSKAR")).
hero_ability(ability("LIFE BREAK AGHANIM'S SCEPTER"), hero("HUSKAR")).
hero_ability(ability("SUN STRIKE"), hero("INVOKER")).
hero_ability(ability("SUN STRIKE AGHANIM'S SCEPTER"), hero("INVOKER")).
hero_ability(ability("ICE PATH"), hero("JAKIRO")).
hero_ability(ability("MACROPYRE"), hero("JAKIRO")).
hero_ability(ability("OMNISLASH"), hero("JUGGERNAUT")).
hero_ability(ability("DUEL"), hero("LEGION COMMANDER")).
hero_ability(ability("LAGUNA BLADE"), hero("LINA")).
hero_ability(ability("FINGER OF DEATH"), hero("LION")).
hero_ability(ability("REVERSE POLARITY"), hero("MAGNUS")).
hero_ability(ability("WUKONG'S COMMAND"), hero("MONKEY KING")).

% ability_type(ability_name, ability_type), where ability type is one of [dmg_single, dmg_aoe, stun_single, stun_aoe]
% dmg_single - huge damage to solo target
% dmg_aoe - huge damage to big area
% stun_single - great control to one target
% stun_single - great control to big area
ability_type(ability("BERSERKER'S CALL"), stun_aoe).
ability_type(ability("FIEND'S GRIP"), stun_single).
ability_type(ability("FLAMING LASSO"), stun_single).
ability_type(ability("PRIMAL ROAR"), stun_single).
ability_type(ability("FREEZING FIELD"), dmg_aoe).
ability_type(ability("VACUUM"), stun_aoe).
ability_type(ability("BEDLAM"), dmg_single).
ability_type(ability("TERRORIZE"), stun_aoe).
ability_type(ability("MAGNETIZE"), dmg_aoe).
ability_type(ability("ECHO SLAM"), stun_aoe).
ability_type(ability("ECHO SLAM"), dmg_aoe).
ability_type(ability("ECHO STOMP"), stun_aoe).
ability_type(ability("EARTH SPLITTER"), dmg_aoe).
ability_type(ability("IMPETUS"), dmg_single).
ability_type(ability("BLACK HOLE"), stun_aoe).
ability_type(ability("BLACK HOLE"), dmg_aoe).
ability_type(ability("CHRONOSPHERE"), stun_aoe).
ability_type(ability("CALL DOWN"), dmg_aoe).
ability_type(ability("SHARPSHOOTER"), dmg_single).
ability_type(ability("LIFE BREAK"), dmg_single).
ability_type(ability("LIFE BREAK AGHANIM'S SCEPTER"), stun_single).
ability_type(ability("SUN STRIKE"), dmg_single).
ability_type(ability("SUN STRIKE AGHANIM'S SCEPTER"), dmg_aoe).
ability_type(ability("ICE PATH"), stun_aoe).
ability_type(ability("MACROPYRE"), dmg_aoe).
ability_type(ability("OMNISLASH"), dmg_aoe).
ability_type(ability("OMNISLASH"), dmg_single).
ability_type(ability("DUEL"), stun_single).
ability_type(ability("LAGUNA BLADE"), dmg_single).
ability_type(ability("FINGER OF DEATH"), dmg_single).
ability_type(ability("REVERSE POLARITY"), stun_aoe).
ability_type(ability("WUKONG'S COMMAND"), dmg_aoe).

% Get hero hero_ability types
hero_ability_type(HERO, TYPE) :- hero_ability(ability(ABILITY), hero(HERO)), ability_type(ability(ABILITY), TYPE).
hero_with_stun(HERO) :- hero_ability_type(HERO, TYPE), (TYPE = stun_single; TYPE = stun_aoe).
hero_with_dmg(HERO) :- hero_ability_type(HERO, TYPE), (TYPE = dmg_single; TYPE = dmg_aoe).
% Hero with both stun skill and dmg skill
universal_hero(HERO) :- hero_with_dmg(HERO), hero_with_stun(HERO).
