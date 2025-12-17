<div align="center">
<h1>XAI-Guided-CoT</h1>
[📖 Tutorial](tutorial.ipynb) | [📊 Results](experiments/results.ipynb)
</div>

## Aim

To orchestrate an end-to-end generate-AI workflow (or pipeline) for automating the Chain-of-Thought (CoT) prompting for tabular binary classification by generating intermediate reasoning from global and local feature importances. 

<div align="center">
  <img src="images/genai_workflow.png" alt="XAI-Guided-CoT Pipeline" width="800">
</div>

<b><u>Note:</u></b> The XAI attributes are obtained by training and tuning a tree-based explainable model and then extracting its feature importances (using sklearn's `feature_importances_` attribute) and SHAP values (using `TreeExplainer`)

## Environment Setup

- Clone this repository.

    ```bash
    git clone https://github.com/Gaurav0502/xai-guided-cot.git
    ```

- Create a virtual environment to isolate all the dependencies from any global dependencies on your local system.

    ```bash
    python3 -m venv venv
    ```

- Activate the virtual environment.

    ```bash
    source venv/bin/activate # MacOS

    venv\Scripts\activate # Windows
    ```

- Install all packages in the ```requirements.txt``` file.

    ```bash
    pip3 install -r requirements.txt
    ```

- Optionally, you may have install `toon_format` from `toon-python` separately. (it is there is also there in the `requirements.txt`)

    ```bash
    pip3 install git+https://github.com/toon-format/toon-python.git
    ```

- Run the `setup.sh` to ensure the required directory structure is created.

    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

- After setting up the dependencies, install the google cloud SDK and authenticate your environment to access Google Cloud Storage (GCS) and Vertex AI.

    ```bash
    gcloud auth application-default login
    ```

- Get API keys from Together AI (`TOGETHER_API_KEY`) and Anthropic Developer Platform (`CLAUDE_API_KEY`). Add them into the `.env` file created by `setup.sh`.

- Use may require a `WANDB_API_KEY` and `WANDB_PROJECT_NAME` since the tree-based `ExplanableModel` is trained and tuned using `wandb sweep`

- Now, you must create setup Google Cloud Provider (GCP) with the following steps:
    1. Create a Project in GCP and record the `PROJECT_ID`.
    2. Inside the Project, create GCP Bucket to store the batch inference job JSONL files for Vertex AI.
    3. Record your `BUCKET_NAME`, `LOCATION`. 
    4. Ensure that the `LOCATION` you choose has the model you want to use because the code in this repository requires the location for both GCP Bucket and Vertex AI Batch Inference to be same (ideally, you can use different ones).

- Finally, your `.env` file must look as follows:

    ```bash
    # wandb config
    WANDB_API_KEY=<YOUR-API-KEY>
    WANDB_PROJECT_NAME=<YOUR-WANDB-PROJECT-NAME>

    # gcp config
    PROJECT_ID=<YOUR-PROJECT-ID>
    BUCKET_NAME=<YOUR-GCP-BUCKET-NAME>
    LOCATION=<YOUR-LOCATION> # same for storage and batch inference

    # together ai config
    TOGETHER_API_KEY=<YOUR-TOGETHERAI-API-KEY>

    # anthropic config
    CLAUDE_API_KEY=<YOUR-CLAUDE-API-KEY>
    ```

<b><u>Note:</u></b> Only the API keys are secrets. Others are just kept inside the `.env` since they define the environment for different SDKs.

- If you wish to check if your environment setup is complete, you can run the unit tests inside `test/` using `pytest`.

    ```bash
    pytest -v
    ```

All tests are expected to be successful if the environment is correctly configured.

## Tutorial

- The repository has a `tutorial.ipynb` file that explains the environment setup procedure in much more detail. 

- It also explains how to use the overall pipeline and its individual components.

## Running the experiments

- We tested the pipeline functionality and prediction performance on four datasets and attempted to answer three research questions:

    1. Can large language models (LLMs) effectively generate natural language reasoning from numerical XAI attributes?

    2. Does providing this natural language reasoning help improve the performance of standard prompt engineering techniques on the tabular binary classification?

    3. How does XAI-Guided-CoT perform in comparison to the tree-based explainable model?

- Additionally, we also attempted to perform two ablation studies to further diagnose the improvement in prediction performance:

    1. Does the improvement of XAI-Guided-CoT over the zero shot baseline happen to be because of CoT alone?
    
    2. Does the semantic context provided by dataset metadata (dataset name, column name, and class names) drive the performance or it is the XAI attributes.

To run these experiments, you can use the `main.py` file in the root of this repository. Preferably, use it inside the terminal because batch inference jobs can take significant amount of time complete.

    ```bash
    python3 main.py --dataset <dataset-name> # without masking dataset metadata

    python3 main.py --dataset <dataset-name> --masked # with masking dataset metadata
    ```
<b><u>Note:</u></b> The dataset name can be one among this list: `titanic`, `loan`, `diabetes`, `mushroom`. Any other dataset name will raise a `ValueError`.


## Results

- To see our results, refer `experiments/results.ipynb`. 

- It includes how the results were interpreted. All batch inference jobs were executed through the terminal.

## References

1. https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini (Vertex AI Batch Inference)

2. https://docs.together.ai/docs/batch-inference (Together AI Batch Job)

3. https://platform.claude.com/docs/en/build-with-claude/batch-processing (Anthropic API Batch Processing)
