import json
import os

from dotenv import load_dotenv
from groq import Groq

from betamark import arc_agi

load_dotenv()


EVAL_CHALELNGES_FILEPATH = "data/arc-agi_evaluation_challenges.json"
eval_challenges_json = json.load(open(EVAL_CHALELNGES_FILEPATH, "r"))


def few_shot_cerebras_predict(input_dict):
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

    # print(PROMPT)

    client = Groq(
        # This is the default and can be omitted
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # try:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        model="llama-3.1-8b-instant",
        max_tokens=8192,
    )
    print(chat_completion)
    # raise KeyError
    print("test")
    # except ZeroDivisionError:
    #     return [[0]]

    print("######")
    print(chat_completion.choices[0].message.content)
    # raise KeyError
    list_repr_string = chat_completion.choices[0].message.content
    list_repr_string = (
        list_repr_string.replace("```json", "")
        .replace("```python", "")
        .replace("```", "")
    )

    try:
        list_repr = eval(list_repr_string)
        print(list_repr)
        # print(type(list_repr))
        return list_repr
    except:
        pass
        print("error!")
        return [[0]]


if __name__ == "__main__":
    results = arc_agi.run_eval(user_func=few_shot_cerebras_predict)
    print(results)
    # few_shot_cerebras_predict([{"train": [0]}])
