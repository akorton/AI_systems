:- use_module(synergy).
:- use_module(abilities).
:- use_module(heroes).

print_test(Test, Expected_result) :- ((Test, Expected_result); (\+ Test, \+ Expected_result)),
    ansi_format([bold, fg(green)], 'Test: ~w passed\n', [Test]).
print_test(Test, Expected_result) :- ansi_format([bold, fg(red)], 'Test: ~w failed\n', [Test]).
print_test(Test, Expected_result, Custom_test_name) :- ((Test, Expected_result); (\+ Test, \+ Expected_result)),
    ansi_format([bold, fg(green)], 'Test: ~w passed\n', [Custom_test_name]).
print_test(Test, Expected_result, Custom_test_name) :- ansi_format([bold, fg(red)], 'Test: ~w failed\n', [Custom_test_name]).

?- writeln("").
?- writeln("Synergy/2 tests.").
?- print_test(heroes_synergy("AXE", "INVOKER"), true). % stun_aoe, dmg_aoe
?- print_test(heroes_synergy("INVOKER", "AXE"), true). % dmg_aoe, stun_aoe
?- print_test(heroes_synergy("BANE", "INVOKER"), true). % stun_single, dmg_single
?- print_test(heroes_synergy("INVOKER", "BANE"), true). % dmg_single, stun_single
?- print_test(heroes_synergy("BATRIDER", "CRYSTAL MAIDEN"), false). % dmg_single, dmg_aoe
?- print_test(heroes_synergy("EARTH SPIRIT", "ENCHANTRESS"), false). % dmg_aoe, dmg_single
?- print_test(heroes_synergy("HOODWINK", "LINA"), false). % dmg_single, dmg_single
?- print_test(heroes_synergy("JUGGERNAUT", "MONKEY KING"), false). % dmg_aoe, dmg_aoe
?- print_test(heroes_synergy("BEASTMASTER", "JAKIRO"), false). % stun_single, stun_aoe
?- print_test(heroes_synergy("DARK SEER", "BANE"), false). % stun_aoe, stun_single
?- print_test(heroes_synergy("BATRIDER", "BEASTMASTER"), false). % stun_single, stun_single
?- print_test(heroes_synergy("DARK WILLOW", "FACELESS VOID"), false). % stun_aoe, stun_aoe
?- print_test(heroes_synergy("EARTHSHAKER", "EARTHSHAKER"), false). % Can't be synergy with itself
?- print_test(heroes_synergy("HUSKAR", "HUSKAR"), false). % Can't be synergy with itself
?- writeln("").
?- writeln("Synergy/3 tests.").
?- print_test(heroes_synergy("ENIGMA", "DARK SEER", "JAKIRO"), true). % Some synergy/3 tests
?- print_test(heroes_synergy("ELDER TITAN", "MAGNUS", "CRYSTAL MAIDEN"), true). % Some synergy/3 tests
?- print_test(heroes_synergy("EARTHSHAKER", "ELDER TITAN", "GYROCOPTER"), true). % Some synergy/3 tests
?- print_test(heroes_synergy("DARK WILLOW", "BANE", "HUSKAR"), true). % Some synergy/3 tests
% Check for synergy with specific hero which should have stun, dmg ability or both
?- writeln("").
?- writeln("Complex synergy tests tests.").
?- print_test((findall(H, (hero(H), heroes_synergy(H, "INVOKER"), universal_hero(H)), L), member("HUSKAR", L)), true, "Universal hero synergy with INVOKER").
?- print_test((findall(H, (hero(H), heroes_synergy(H, "GYROCOPTER"), universal_hero(H)), L), member("ENIGMA", L)), true, "Universal hero synergy with GYROCOPTER").
?- print_test((findall(H, (hero(H), heroes_synergy(H, "BANE"), hero_with_stun(H)), L), member("DARK WILLOW", L)), true, "Hero with stun synergy with BANE").
?- print_test((findall(H, (hero(H), heroes_synergy(H, "JUGGERNAUT"), hero_with_stun(H)), L), member("BATRIDER", L)), true, "Hero with stun synergy with JUGGERNAUT").
?- print_test((findall(H, (hero(H), heroes_synergy(H, "LION"), hero_with_dmg(H)), L), member("HUSKAR", L)), true, "Hero with damage synergy with LION").
?- print_test((findall(H, (hero(H), heroes_synergy(H, "MONKEY KING"), hero_with_dmg(H)), L), member("JAKIRO", L)), true, "Hero with damage synergy with MONKEY KING").
?- writeln("").
