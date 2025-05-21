# ECB CRC web backend

## Description
This project is a web backend service for colorectal cancer (CRC) prediction. The service provides a REST API that accepts gene expression data and returns cancer risk predictions.

## API
```shell
curl -X POST "http://10.10.10.111:8211/crc/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [
    {
        "SampleID": "test_sample_1",
        "GAPDH": 22,
        "RB012": 32,
        "RB018": 34,
        "RB020": 15,
        "RB054": 26.6,
        "RB080": 37.4,
        "RB102": 27,
        "RB117": 19,
        "RB167": 21.2
    },
    {
        "SampleID": "test_sample_2",
        "GAPDH": "e",
        "RB012": 13,
        "RB018": "e",
        "RB020": 21,
        "RB054": 35,
        "RB080": "e",
        "RB102": 24,
        "RB117": 15.1,
        "RB167": 14
    },
    {
        "SampleID": "test_sample_3",
        "GAPDH": 29,
        "RB012": 31.2,
        "RB018": 45,
        "RB020": 32.4,
        "RB054": 35.5,
        "RB080": 37.4,
        "RB102": 27.5,
        "RB117": 19.1,
        "RB167": 21.2
    },
    {
        "SampleID": "test_sample_4",
        "GAPDH": 29,
        "RB012": 31.2,
        "RB018": 28.2,
        "RB020": "",
        "RB054": 35.5,
        "RB080": 37.4,
        "RB102": 27.5,
        "RB117": 19.1,
        "RB167": 21.2
    },
    {
        "SampleID": "test_sample_5",
        "GAPDH": 22.3,
        "RB012": 31.2,
        "RB018": 31.2,
        "RB020": "Undetermined",
        "RB054": 35.5,
        "RB080": 37.4,
        "RB102": 27.5,
        "RB117": 19.1,
        "RB167": 21.2
    }
]}'
```

+ Output
```
[
    {
        "SampleID": "test_sample_1",
        "Predict Score": 0.7824867725,
        "Result": "High risk"
    },
    {
        "SampleID": "test_sample_5",
        "Predict Score": 0.8452777778,
        "Result": "High risk"
    },
    {
        "SampleID": "test_sample_3",
        "Predict Score": "",
        "Result": "INVALID DATA"
    },
    {
        "SampleID": "test_sample_2",
        "Predict Score": "",
        "Result": "INCOMPLETE DATA"
    },
    {
        "SampleID": "test_sample_4",
        "Predict Score": "",
        "Result": "INCOMPLETE DATA"
    }
]
```