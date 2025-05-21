import os
import sys
import json
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, scripts_dir)

from MLGenie_prediction import MLGeniePredictor

data_json = os.path.join(scripts_dir, "test/test.json")
data_error_json = os.path.join(scripts_dir, "test/test_error.json")

test_data = json.load(open(data_json))
test_error_data = json.load(open(data_error_json))

predictor = MLGeniePredictor(test_data)
pred_result = predictor.run()
print(pred_result)
with open(os.path.join(scripts_dir, "test/predict_result.json"), "w") as f:
    f.write(pred_result)

predictor_error = MLGeniePredictor(test_error_data)
print(predictor_error.run())


