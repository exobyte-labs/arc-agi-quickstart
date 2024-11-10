import os
import json
import tqdm
import itertools

from betamark import arc_agi

from random import sample

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

test_challenges_path = "data/arc-agi_test_challenges.json"
train_challenges_path = "data/arc-agi_training_challenges.json"
train_solutions_path = "data/arc-agi_training_solutions.json"

with open(test_challenges_path) as fp:
    test_challenges = json.load(fp)
with open(train_challenges_path) as fp:
    train_challenges = json.load(fp)
with open(train_solutions_path) as fp:
    train_solutions = json.load(fp)


def plot_task(task):
    """plots a task"""
    examples = task["train"]
    n_examples = len(examples)
    cmap = ListedColormap(
        [
            "#000",
            "#0074D9",
            "#FF4136",
            "#2ECC40",
            "#FFDC00",
            "#AAAAAA",
            "#F012BE",
            "#FF851B",
            "#7FDBFF",
            "#870C25",
        ]
    )
    norm = Normalize(vmin=0, vmax=9)
    figure, axes = plt.subplots(2, n_examples, figsize=(n_examples * 4, 8))
    for column, example in enumerate(examples):
        axes[0, column].imshow(example["input"], cmap=cmap, norm=norm)
        axes[1, column].imshow(example["output"], cmap=cmap, norm=norm)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    plt.show()


# defining a handful of basic primitives


def tophalf(grid):
    """upper half"""
    return grid[: len(grid) // 2]


def rot90(grid):
    """clockwise rotation by 90 degrees"""
    return list(zip(*grid[::-1]))


def hmirror(grid):
    """mirroring along horizontal"""
    return grid[::-1]


def compress(grid):
    """removes frontiers"""
    ri = [i for i, r in enumerate(grid) if len(set(r)) == 1]
    ci = [j for j, c in enumerate(zip(*grid)) if len(set(c)) == 1]
    return [
        [v for j, v in enumerate(r) if j not in ci]
        for i, r in enumerate(grid)
        if i not in ri
    ]


def trim(grid):
    """removes border"""
    return [r[1:-1] for r in grid[1:-1]]


# defining the DSL as the set of the primitives

DSL_primitives = {tophalf, rot90, hmirror, compress, trim}
primitive_names = {p.__name__ for p in DSL_primitives}
print(f"DSL consists of {len(DSL_primitives)} primitives: {primitive_names}")


# the maximum composition depth to consider
MAX_DEPTH = 6

# construct the program strings of all programs expressible by composing at most MAX_DEPTH primitives

program_strings = []
for depth in range(1, MAX_DEPTH + 1):
    primitive_tuples = itertools.product(*[primitive_names] * depth)
    for primitives in primitive_tuples:
        left_side = "".join([p + "(" for p in primitives])
        right_side = ")" * depth
        program_string = f"lambda grid: {left_side}grid{right_side}"
        program_strings.append(program_string)


# print some of the program strings
print(f"Space to search consists of {len(program_strings)} programs:\n")
print("\n".join([*program_strings[:10], "..."]))


# map program strings to programs
programs = {prog_str: eval(prog_str) for prog_str in program_strings}


# for each task, search over the programs and if a working program is found, remember it

guesses = dict()
# iterate over all tasks
for key, task in tqdm.tqdm(train_challenges.items()):
    train_inputs = [example["input"] for example in task["train"]]
    train_outputs = [example["output"] for example in task["train"]]
    hypotheses = []
    # iterate over all programs
    for program_string, program in programs.items():
        try:
            if all([program(i) == o for i, o in zip(train_inputs, train_outputs)]):
                # remember program if it explains all training examples
                hypotheses.append(program_string)
        except:
            pass
    # select first program for making predictions
    if len(hypotheses) > 0:
        print(f"found {len(hypotheses)} candidate programs for task {key}!")
        guesses[key] = hypotheses[0]
print(f"\nMade guesses for {len(guesses)} tasks")

# make predictions and evaluate them

solved = dict()

# iterate over all tasks for which a guess exists
for key, program_string in guesses.items():
    test_inputs = [example["input"] for example in train_challenges[key]["test"]]
    program = eval(program_string)
    if all([program(i) == o for i, o in zip(test_inputs, train_solutions[key])]):
        # mark predition as correct if all test examples are solved by the program
        solved[key] = program_string


print(f"Predictions correct for {len(solved)}/{len(guesses)} tasks")


# visualize solved tasks
for key, program_string in solved.items():
    print(f'For task "{key}", found program "{program_string}"')
    plot_task(train_challenges[key])


# let's try to make a submission

submission = dict()
# iterate over all tasks
for key, task in tqdm.tqdm(test_challenges.items()):
    train_inputs = [example["input"] for example in task["train"]]
    train_outputs = [example["output"] for example in task["train"]]
    hypotheses = []
    # iterate over all programs
    for program_string, program in programs.items():
        try:
            if all([program(i) == o for i, o in zip(train_inputs, train_outputs)]):
                # remember program if it explains all training examples
                hypotheses.append(program_string)
        except:
            pass
    # select first program for making predictions
    predictions = [example["input"] for example in task["test"]]
    if len(hypotheses) > 0:
        print(f"found {len(hypotheses)} candidate programs for task {key}!")
        program_string = hypotheses[0]
        program = eval(program_string)
        try:
            predictions = [program(example["input"]) for example in task["test"]]
        except:
            pass
    # print(predictions[0])
    submission[key] = [{"attempt_1": grid, "attempt_2": grid} for grid in predictions]
print(f"\nMade guesses for {len(guesses)} tasks")


def make_dsl_prediction(task):
    train_inputs = [example["input"] for example in task["train"]]
    train_outputs = [example["output"] for example in task["train"]]
    hypotheses = []
    # iterate over all programs
    for program_string, program in programs.items():
        try:
            if all([program(i) == o for i, o in zip(train_inputs, train_outputs)]):
                # remember program if it explains all training examples
                hypotheses.append(program_string)
        except:
            pass
    # select first program for making predictions
    predictions = [example["input"] for example in task["test"]]
    if len(hypotheses) > 0:
        print(f"found {len(hypotheses)} candidate programs for task {key}!")
        program_string = hypotheses[-1]
        program = eval(program_string)
        try:
            predictions = [program(example["input"]) for example in task["test"]]
            return predictions[0]
        except:
            return [[0]]


result = arc_agi.run_eval(user_func=make_dsl_prediction)
print(result)

# with open("submission.json", "w") as fp:
#     json.dump(submission, fp)
