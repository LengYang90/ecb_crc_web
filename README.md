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

3. Use the execute API:
```shell
# Only for local environment
curl --location --request POST 'http://127.0.0.1:2025/bagenie/executelocal' \
--header 'Content-Type: application/json' \
--data-raw '{
    "goal": "The rows of the matrix are the genes and the columns are the samples. For host gene expression data for each disease cohort. We filtered out lowly expressed genes to retain genes that are expressed in at least half of the samples in each disease cohort.  We performed variance stabilizing transformation using the R package '\''DESeq2'\'' on the filtered gene expression read count data.  We filtered out genes with low variance, using 25% quantile of variance across samples in each disease cohort as cut-off.  Performing these steps for RNA-seq data for each disease cohort separately resulted in a unique host gene expression matrix per disease for downstream analysis.",
    "outdir": "/mnt/data/lengyang/youjia_project/autoba/res",
    "chatbot_model": "gpt-4o",
    "chatbot_key": "",
    "docker_env" : "lengyang/pb_bio_tools:1.0.1",
    "max_try_num" : 5,
    "user_expertise_level": "intermediate"
}'
```

### Usage Docker

1. Plan
```Shell
# docker build
docker build -t bagenie_plan:1.1.0 .

# docker run
docker run \
-e "AWS_ACCESS_KEY=XXX" \
-e "AWS_SECRET_KEY=XXX" \
-e "AWS_REGION=XXX"  \
-p 2025:2025 bagenie_plan:1.1.0
```


2. Execute
```Shell
# docker build
docker build -t bagenie_exe:1.1.0 -f scripts/execute_docker/Dockerfile .

# docker run
docker run bagenie_exe:1.1.0
```


### Usage Commnad Line

1. Use the plan command:
```python
from scripts.bagenie_plan import BAGeniePlan


bagenie_plan = BAGeniePlan(
    input_config = input_json_data,
    goal = goal,
    outdir = outdir,
    chatbot_model = chatbot_model,
    chatbot_key = chatbot_key,
    user_expertise_level = "intermediate"
    )
bagenie_plan.bagenie_plan()
```

2. Use the execute command:
```python
from scripts.bagenie_execute import BAGenieExecute

bagenie_execute = BAGenieExecute(
    goal = goal,
    outdir = outdir,
    chatbot_model = chatbot_model,
    chatbot_key = chatbot_key,
    docker_env = "lengyang/pb_bio_tools:1.0.1",
    max_try_num = 5,
    user_expertise_level = "intermediate"
    )
bagenie_execute.bagenie_execute()
```

### Output

1. Plan
```Shell
└── outdir
    ├── execute_plan.json # [ input file description data, response for plan, parameters for plan ]
    └── prompt.txt # prompt for plan
```

2. Execute
```Shell
└── outdir
    ├── execute_plan.json
    ├── execute_log.json # execute log data
    ├── Step_1_Output # execute work dir
    │   ├── command_0.sh
    │   ├── command_success.sh
    │   └── ...
    ├── Step_2_Output
    ├── ...
```