from pyswip import Prolog
from typing import List, Tuple, Dict

# region other setup
heroes_types: List[str] = ["со станом", "с уроном", "универсального"]
types_to_prolog_functions: Dict[str, str] = {"со станом": "hero_with_stun", "с уроном": "hero_with_dmg", "универсального": "universal_hero"}
default_retry_message = "Повторите ввод."
# endregion

# region prolog setup
prolog = Prolog()
prolog.consult("synergy.pl")
prolog.consult("heroes.pl")
prolog.consult("abilities.pl")


def synergy_2(hero: str):
    return list(prolog.query(f"hero(X), heroes_synergy(\"{hero}\", X)"))


def synergy_3(hero1: str, hero2: str):
    return list(prolog.query(f"hero(X), heroes_synergy(\"{hero1}\", \"{hero2}\", X)"))


def synergy_with_type_2(hero_type: str, hero: str):
    return list(prolog.query(f"hero(X), {types_to_prolog_functions[hero_type]}(X), heroes_synergy(\"{hero}\", X)"))


def synergy_with_type_3(hero_type: str, hero1: str, hero2: str):
    return list(prolog.query(f"hero(X), {types_to_prolog_functions[hero_type]}(X), heroes_synergy(\"{hero1}\", \"{hero2}\", X)"))


# endregion

# region templates setup

# Each template is {template_text, list_of_lists} and in each list are possibilities for each position
templates: Tuple[str, List[List[str]]] = [
    ("Какой герой хорош в связке с _", [[]]),
    ("Какой герой хорош в связке с _ и _", [[], []]),
    ("Я хочу пикнуть героя _ в связке с _", [heroes_types, []]),
    ("Я хочу пикнуть героя _ в связке с _ и _", [heroes_types, [], []]),
]
template_handlers = [synergy_2, synergy_3, synergy_with_type_2, synergy_with_type_3]
# endregion


def print_templates():
    for idx, temp in enumerate(templates, 1):
        print(f"{idx}. {temp[0]}")


def template_with_pos_numbers(template_text: str):
    cnt = 1
    while "_" in template_text:
        template_text = template_text.replace("_", f"${cnt}", 1)
        cnt += 1
    return template_text


def input_for_each_space_in_template(choices: List[List[str]]):
    answers: List[str] = []
    for idx, choice in enumerate(choices):
        while True:
            if len(choice) != 0:
                print("Возможные варианты для данного пропуска:")
                for i in choice:
                    print(i)
            cur_in = input(f"Ввод для пропуска ${idx + 1}: ")
            if len(choice) == 0 or cur_in in choice:
                answers.append(cur_in)
                break
            else:
                print(default_retry_message)
                continue
    return answers


def flatten_prolog_answer(answer: List[Dict[str, str]]):
    return list(set([i["X"].decode("utf-8") for i in answer]))


def handle_prolog_answer(answer: List[str]):
    if len(answer) == 0:
        print("По данному запросу не нашлось результатов.")
        return
    if len(answer) == 1:
        print(f"Нашелся единственный ответ: {answer[0]}")
    else:
        print(f"Нашлось {len(answer)} ответов.")
        while True:
            number = input("Сколько из них вывести: ")
            try:
                number = int(number)
            except ValueError:
                print(default_retry_message)
                continue
            if number < 1 or number > len(answer):
                print(default_retry_message)
                continue
            break
        for i in range(int(number)):
            print(answer[i])


def handle_input():
    print_templates()
    number = input("Введите номер шаблона для поиска в базе знаний или 0 для выхода: \n")
    try:
        number = int(number)
    except ValueError:
        print(default_retry_message)
        return False
    if number == 0:
        return True
    if number < 1 or number > len(templates):
        print(default_retry_message)
        return False

    cur_template = templates[number - 1]
    print(template_with_pos_numbers(cur_template[0]))
    answers = input_for_each_space_in_template(cur_template[1])
    prolog_answer = template_handlers[number - 1](*answers)
    handle_prolog_answer(flatten_prolog_answer(prolog_answer))

    return True


if __name__ == "__main__":
    exit_fl = False
    while True:
        exit_fl = handle_input()
        if exit_fl:
            print("Хорошего дня!")
            break
