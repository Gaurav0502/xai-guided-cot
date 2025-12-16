# modules used for data handling
import pandas as pd
import json

# modules used for handling
import os

# modules used for testing
import pytest
import tempfile

# modules to be tested
from scripts.configs import Model, COT, Dataset

# type hints
from typing import Generator

# fixtures

# dummy preprocessing 
# function
@pytest.fixture
def dummy_preprocess_fn():
    """Dummy preprocessing function"""

    def preprocess(df):
        return df

    return preprocess

# temporary CSV 
# file
@pytest.fixture
def temp_csv_file() -> Generator[str, None, None]:
    """Create temporary CSV file"""

    # create a temporary 
    # CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # sample data
        df = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
        df.to_csv(f.name, index=False)
        yield f.name

    # remove the 
    # temporary file
    os.unlink(f.name)


# temporary JSON 
# file
@pytest.fixture
def temp_json_file() -> Generator[str, None, None]:
    """Create temporary JSON config file"""

    # sample config 
    # data
    config_data = {
        "dataset_path": "test.csv",
        "shap_values_path": "shap.csv",
        "feature_importances": {"feature1": 0.6},
        "train_data_idx": [0],
        "test_data_idx": [1],
        "train_predictions": [0],
        "test_predictions": [1]
    }

    # create a temporary 
    # JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        yield f.name

    # remove the temporary 
    # JSON file
    os.unlink(f.name)


# temperature validation 
# tests
class TestModelTemperature:
    """Tests for Model temperature validation"""
    
    # valid temperature case
    def test_temperature_valid_zero(self) -> None:
        """Test valid temperature = 0.0 (boundary case)"""

        model = Model(
            provider="google",
            name="gemini-2.5-flash",
            temperature=0.0,
            max_tokens=100
        )

        assert model.temperature == 0.0
    
    # negative temperature 
    # (invalid) case
    def test_temperature_invalid_negative_small(self) -> None:
        """Test invalid temperature = -0.1"""

        with pytest.raises(ValueError, match="Temperature must be >= 0"):
            Model(
                provider="google",
                name="gemini-2.5-flash",
                temperature=-0.1,
                max_tokens=100
            )


# max_tokens validation 
# tests
class TestModelMaxTokens:
    """Tests for Model max_tokens validation"""
    
    # valid max_tokens case
    def test_max_tokens_valid_minimum(self) -> None:
        """Test valid max_tokens = 1 (minimum valid)"""

        model = Model(
            provider="google",
            name="gemini-2.5-flash",
            temperature=0.7,
            max_tokens=1
        )

        assert model.max_tokens == 1
    
    # zero max_tokens (invalid) case
    def test_max_tokens_invalid_zero(self) -> None:
        """Test invalid max_tokens = 0 (boundary case)"""

        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            Model(
                provider="google", 
                name="gemini-2.5-flash",
                temperature=0.7,
                max_tokens=0 
            )

    # negative max_tokens (invalid) case
    def test_max_tokens_invalid_negative(self) -> None:
        """Test invalid max_tokens = -1"""
        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            Model(
                provider="google",
                name="gemini-2.5-flash",
                temperature=0.7,
                max_tokens=-1
            )


# thinking_budget validation 
# tests
class TestCOTThinkingBudget:
    """Tests for COT thinking_budget validation"""
    
    # valid thinking_budget case
    def test_thinking_budget_valid_zero(self) -> None:
        """Test valid thinking_budget = 0 (boundary case)"""

        cot = COT(
            num_examples_per_agent=10,
            reasoning={},
            thinking_budget=0
        )

        assert cot.thinking_budget == 0
    
    # negative thinking_budget (invalid) case
    def test_thinking_budget_invalid_negative_small(self) -> None:
        """Test invalid thinking_budget = -1"""

        with pytest.raises(ValueError, match="thinking_budget must be >= 0"):
            COT(
                num_examples_per_agent=10,
                reasoning={},
                thinking_budget=-1
            )


# dataset path file extension 
# tests
class TestDatasetPathExtension:
    """Tests for Dataset path file extension validation"""
    
    # valid CSV extension case
    def test_path_valid_csv_extension(
            self, 
            temp_csv_file, 
            temp_json_file, 
            dummy_preprocess_fn
        ) -> None:
        """Test valid path with .csv extension"""

        dataset = Dataset(
            name="test",
            path=temp_csv_file,
            config_file_path=temp_json_file,
            shap_vals_path="shap.csv",
            preprocess_fn=dummy_preprocess_fn,
            target_col="target",
            labels={0: "class0", 1: "class1"}
        )

        assert dataset.path.endswith(".csv")
    
    # invalid JSON extension case
    def test_path_invalid_json_extension(
            self, 
            temp_json_file, 
            dummy_preprocess_fn
        ) -> None:
        """Test invalid path with .json extension"""

        with pytest.raises(ValueError, match="must end with '.csv'"):
            Dataset(
                name="test",
                path="data.json",  # Wrong extension
                config_file_path=temp_json_file,
                shap_vals_path="shap.csv",
                preprocess_fn=dummy_preprocess_fn,
                target_col="target",
                labels={0: "class0", 1: "class1"}
            )
    
    # invalid no extension case
    def test_path_invalid_no_extension(
            self, 
            temp_json_file, 
            dummy_preprocess_fn
        ) -> None:
        """Test invalid path with no extension"""

        with pytest.raises(ValueError, match="must end with '.csv'"):
            Dataset(
                name="test",
                path="data",
                config_file_path=temp_json_file,
                shap_vals_path="shap.csv",
                preprocess_fn=dummy_preprocess_fn,
                target_col="target",
                labels={0: "class0", 1: "class1"}
            )
    
    # invalid empty string case
    def test_path_invalid_empty_string(
            self, 
            temp_json_file, 
            dummy_preprocess_fn
        ) -> None:
        """Test invalid path with empty string"""

        # empty string (invalid) case
        with pytest.raises(ValueError, match="must end with '.csv'"):
            Dataset(
                name="test",
                path="",
                config_file_path=temp_json_file,
                shap_vals_path="shap.csv",
                preprocess_fn=dummy_preprocess_fn,
                target_col="target",
                labels={0: "class0", 1: "class1"}
            )


# dataset path file existence 
# tests
class TestDatasetPathExistence:
    """Tests for Dataset path file existence validation"""
    
    # valid existing file case
    def test_path_valid_existing_file(
            self, 
            temp_csv_file, 
            temp_json_file, 
            dummy_preprocess_fn
        ) -> None:
        """Test valid path to existing CSV file"""

        dataset = Dataset(
            name="test",
            path=temp_csv_file,
            config_file_path=temp_json_file,
            shap_vals_path="shap.csv",
            preprocess_fn=dummy_preprocess_fn,
            target_col="target",
            labels={0: "class0", 1: "class1"}
        )

        assert os.path.isfile(dataset.path)
    
    # invalid nonexistent file case
    def test_path_invalid_nonexistent_file(
            self, 
            temp_json_file, 
            dummy_preprocess_fn
        ):
        """Test invalid path to non-existent file"""

        # nonexistent file (invalid) case
        with pytest.raises(FileNotFoundError):
            Dataset(
                name="test",
                path="nonexistent_file.csv",
                config_file_path=temp_json_file,
                shap_vals_path="shap.csv",
                preprocess_fn=dummy_preprocess_fn,
                target_col="target",
                labels={0: "class0", 1: "class1"}
            )

