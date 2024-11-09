import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from betamark import arc_agi

load_dotenv()

EVAL_CHALELNGES_FILEPATH = "data/arc-agi_evaluation_challenges.json"
eval_challenges_json = json.load(open(EVAL_CHALELNGES_FILEPATH, "r"))


def zero_shot_gpt_predict(input_dict):
    PROMPT = """
    What is the output to this sequence?

    Only answer as a double list

    """
    for i in range(len(input_dict["train"])):
        input = input_dict["train"][i]["input"]
        PROMPT += f'\n"input": {input}'
        output = input_dict["train"][i]["output"]
        PROMPT += f'\n"output": {output}'

    PROMPT += f'\n"input": {input_dict["test"][0]["input"]}'

    print(PROMPT)

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY_ARC"),
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": PROMPT,
                }
            ],
            model="gpt-4o-mini",
        )
    except:
        return [[0]]

    print("######")
    print(chat_completion.choices[0].message.content)
    list_repr_string = chat_completion.choices[0].message.content
    list_repr_string = (
        list_repr_string.replace("```json", "")
        .replace("```python", "")
        .replace("```", "")
    )

    try:
        list_repr = eval(list_repr_string)
        print(list_repr)
        print(type(list_repr))
        return list_repr
    except:
        pass
        print("error!")
        return [[0]]

    return [[0]]


if __name__ == "__main__":
    results = arc_agi.run_eval(user_func=zero_shot_gpt_predict)
    print(results)
