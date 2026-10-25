import pickle
import sklearn
import os
import json
from pathlib import Path
from urllib.request import Request, urlopen

'''Write a Python script that loads `pipeline.bin` with `pickle` and calls
`predict_proba` for this lead:'''
    

lead = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}

MODEL_PATH = Path(Path(__file__).with_name("pipeline.bin"))
with MODEL_PATH.open("rb") as model_file:
    pipeline = pickle.load(model_file)

'''Report the probability of conversion rounded to three decimal places. The
grader accepts an absolute error of at most `0.005`; do not choose the nearest
answer from a list.'''
result = pipeline.predict_proba([lead])[0, 1]
print(result)
print("----")
print(round(pipeline.predict_proba([lead])[0, 1], 3))


#{"conversion_probability":0.769799,"conversion":true}✔
'''Report `conversion_probability` rounded to three decimal places. The same
`0.005` absolute tolerance applies. The response must contain a probability in
`[0, 1]` and a boolean `conversion` field.'''
client = {
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74
}
request = Request(
    "http://localhost:9696/predict",
    data=json.dumps(client).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urlopen(request) as response:
    result = json.load(response)

assert set(result) == {"conversion_probability", "conversion"}
assert isinstance(result["conversion_probability"], (int, float)) and not isinstance(
    result["conversion_probability"], bool
)
assert 0 <= result["conversion_probability"] <= 1
assert isinstance(result["conversion"], bool)
print(
    f'{{"conversion_probability": {result["conversion_probability"]:.3f}, '
    f'"conversion": {json.dumps(result["conversion"])}}}'
)