:- use_module(synergy).

print_test(Test, Expected_result) :- ((Test, Expected_result); (\+ Test, \+ Expected_result)),
    ansi_format([bold, fg(green)], 'Test: ~w passed\n', [Test]).
print_test(Test, Expected_result) :- ansi_format([bold, fg(red)], 'Test: ~w failed\n', [Test]).

?- writeln("Synergy tests.").
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
