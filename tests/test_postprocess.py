# modules used for data handling
import json
import os

# modules used for testing
import pytest
import tempfile

# modules to be tested
from scripts.postprocess import (
    parse_baseline_llm_results,
    parse_reasoning_llm_results,
    parse_objective_judge_results,
    parse_zero_shot_cot_llm_results,
    parse_cot_llm_results
)


# baseline llm results fixtures
@pytest.fixture
def baseline_jsonl_valid():
    """Valid baseline results"""

    # sample data
    data = [
        {
            "key": "baseline_zero-shot_batch-0",
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "0"}]
                    }
                }]
            }
        },
        {
            "key": "baseline_zero-shot_batch-1",
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "The answer is 1"}]
                    }
                }]
            }
        }
    ]

    # create temporary file
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    try:
        with os.fdopen(fd, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


# baseline llm results 
# edge cases
@pytest.fixture
def baseline_jsonl_edge_cases():
    """Edge cases: interrupted, no numeric, empty text"""

    # sample data
    data = [
        {
            "key": "baseline_zero-shot_batch-0",
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "0"}]
                    }
                }]
            }
        },
        {
            "key": "baseline_zero-shot_batch-1",
            "response": {
                "candidates": [{
                    "finishReason": "MAX_TOKENS",  # Interrupted
                    "content": {
                        "parts": [{"text": "1"}]
                    }
                }]
            }
        },
        {
            "key": "baseline_zero-shot_batch-2",
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "I cannot determine"}]
                    }
                }]
            }
        },
        {
            "key": "baseline_zero-shot_batch-3",
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": ""}]
                    }
                }]
            }
        }
    ]

    # create temporary file
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    try:
        with os.fdopen(fd, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


# reasoning llm results fixtures
@pytest.fixture
def reasoning_jsonl_valid():
    """Valid reasoning results"""

    # sample data
    data = [
        {
            "custom_id": "reasoning_batch-0",
            "response": {
                "body": {
                    "choices": [{
                        "message": {
                            "content": "Some text</think><REASONING>This is the reasoning text</REASONING>"
                        }
                    }]
                }
            }
        },
        {
            "custom_id": "reasoning_batch-1",
            "response": {
                "body": {
                    "choices": [{
                        "message": {
                            "content": "Prefix</think><REASONING>Another reasoning</REASONING>"
                        }
                    }]
                }
            }
        }
    ]

    # create temporary file
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    try:
        with os.fdopen(fd, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)

# reasoning llm results 
# edge cases
@pytest.fixture
def reasoning_jsonl_edge_cases():
    """Edge cases: missing delimiters"""

    # sample data
    data = [
        {
            "custom_id": "reasoning_batch-0",
            "response": {
                "body": {
                    "choices": [{
                        "message": {
                            "content": "Some text</think><REASONING>Valid reasoning</REASONING>"
                        }
                    }]
                }
            }
        },
        {
            "custom_id": "reasoning_batch-1",
            "response": {
                "body": {
                    "choices": [{
                        "message": {
                            "content": "Missing </think> delimiter"
                        }
                    }]
                }
            }
        },
        {
            "custom_id": "reasoning_batch-2",
            "response": {
                "body": {
                    "choices": [{
                        "message": {
                            "content": "Has </think> but missing <REASONING>"
                        }
                    }]
                }
            }
        },
        {
            "custom_id": "reasoning_batch-3",
            "response": {
                "body": {
                    "choices": [{
                        "message": {
                            "content": ""
                        }
                    }]
                }
            }
        }
    ]

    # create temporary file
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    try:
        with os.fdopen(fd, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


# objective judge results 
# valid results
@pytest.fixture
def objective_judge_jsonl_valid():
    """Valid objective judge results"""

    # sample data
    data = [
        {
            "request_id": "judge_batch-0",
            "evaluation": "Some analysis text\nEVALUATION:{\"metrics\":{\"faithfulness\":4.5,\"consistency\":4.0,\"coherence\":4.75}}"
        },
        {
            "request_id": "judge_batch-1",
            "evaluation": "Analysis\nEVALUATION:{\"metrics\":{\"faithfulness\":3.5,\"consistency\":4.5,\"coherence\":4.0}}"
        }
    ]

    # create temporary file
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    try:
        with os.fdopen(fd, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


# objective judge results 
# edge cases
@pytest.fixture
def objective_judge_jsonl_edge_cases():
    """Edge cases: missing delimiter, invalid JSON"""

    # sample data
    data = [
        {
            "request_id": "judge_batch-0",
            "evaluation": "Analysis\nEVALUATION:{\"metrics\":{\"faithfulness\":4.5,\"consistency\":4.0,\"coherence\":4.75}}"
        },
        {
            "request_id": "judge_batch-1",
            "evaluation": "Missing EVALUATION: delimiter"
        },
        {
            "request_id": "judge_batch-2",
            "evaluation": "EVALUATION:{\"metrics\":{\"faithfulness\":4.5,\"consistency\":4.0}}"  # Missing coherence
        },
        {
            "request_id": "judge_batch-3",
            "evaluation": "EVALUATION:{invalid json}"
        },
        {
            "request_id": "judge_batch-4",
            "evaluation": "EVALUATION:{\"wrong_key\":{\"value\":1}}"  # Missing metrics key
        }
    ]

    # create temporary file
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    try:
        with os.fdopen(fd, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


# zero-shot cot llm results 
# valid results
@pytest.fixture
def zero_shot_cot_jsonl_valid():
    """Valid zero-shot CoT results"""

    # sample data
    data = [
        {
            "key": "baseline_zero-shot-cot_batch-0",
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "Thinking...\nFINAL PREDICTION: 0"}]
                    }
                }]
            }
        },
        {
            "key": "baseline_zero-shot-cot_batch-1",
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "Analysis\nFINAL PREDICTION: 1"}]
                    }
                }]
            }
        }
    ]

    # create temporary file
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    try:
        with os.fdopen(fd, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


# zero-shot cot llm results 
# edge cases
@pytest.fixture
def zero_shot_cot_jsonl_edge_cases():
    """Edge cases: interrupted, missing delimiter, no numeric"""

    # sample data
    data = [
        {
            "key": "baseline_zero-shot-cot_batch-0",
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "FINAL PREDICTION: 0"}]
                    }
                }]
            }
        },
        {
            "key": "baseline_zero-shot-cot_batch-1",
            "response": {
                "candidates": [{
                    "finishReason": "MAX_TOKENS",  # Interrupted
                    "content": {
                        "parts": [{"text": "FINAL PREDICTION: 1"}]
                    }
                }]
            }
        },
        {
            "key": "baseline_zero-shot-cot_batch-2",
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "Missing FINAL PREDICTION: delimiter"}]
                    }
                }]
            }
        },
        {
            "key": "baseline_zero-shot-cot_batch-3",
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "FINAL PREDICTION: no number here"}]
                    }
                }]
            }
        }
    ]

    # create temporary file
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    try:
        with os.fdopen(fd, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


# cot llm results 
# valid results
@pytest.fixture
def cot_jsonl_valid():
    """Valid CoT results with multiple agents"""

    # sample data
    data = [
        {
            "key": "cot_batch-0_agent-0",  # row_id=0, agent_id=0
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "FINAL PREDICTION: 0"}]
                    }
                }]
            }
        },
        {
            "key": "cot_batch-0_agent-1",  # row_id=0, agent_id=1
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "FINAL PREDICTION: 0"}]
                    }
                }]
            }
        },
        {
            "key": "cot_batch-0_agent-2",  # row_id=0, agent_id=2
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "FINAL PREDICTION: 1"}]
                    }
                }]
            }
        },
        {
            "key": "cot_batch-1_agent-0",  # row_id=1, agent_id=0
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "FINAL PREDICTION: 1"}]
                    }
                }]
            }
        }
    ]

    # create temporary file
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    try:
        with os.fdopen(fd, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)

# cot llm results 
# edge cases
@pytest.fixture
def cot_jsonl_edge_cases():
    """Edge cases: interrupted, missing delimiter, no numeric, majority vote"""

    # sample data
    data = [
        {
            "key": "cot_batch-0_agent-0",  # row_id=0, agent_id=0
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "FINAL PREDICTION: 0"}]
                    }
                }]
            }
        },
        {
            "key": "cot_batch-0_agent-1",  # row_id=0, agent_id=1
            "response": {
                "candidates": [{
                    "finishReason": "MAX_TOKENS",  # Interrupted
                    "content": {
                        "parts": [{"text": "FINAL PREDICTION: 0"}]
                    }
                }]
            }
        },
        {
            "key": "cot_batch-0_agent-2",  # row_id=0, agent_id=2
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "Missing FINAL PREDICTION:"}]
                    }
                }]
            }
        },
        {
            "key": "cot_batch-0_agent-3",  # row_id=0, agent_id=3
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "FINAL PREDICTION: no number"}]
                    }
                }]
            }
        },
        {
            "key": "cot_batch-1_agent-0",  # row_id=1, agent_id=0
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "FINAL PREDICTION: 1"}]
                    }
                }]
            }
        },
        {
            "key": "cot_batch-1_agent-1",  # row_id=1, agent_id=1
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "FINAL PREDICTION: 1"}]
                    }
                }]
            }
        },
        {
            "key": "cot_batch-1_agent-2",  # row_id=1, agent_id=2
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": "FINAL PREDICTION: 0"}]
                    }
                }]
            }
        }
    ]

    # create temporary file
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    try:
        with os.fdopen(fd, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


# baseline llm results 
# parsing tests
class TestParseBaselineLLMResults:

    # valid results test
    def test_valid_results(self, baseline_jsonl_valid):
        """Test parsing valid baseline results"""

        result = parse_baseline_llm_results(baseline_jsonl_valid)
        assert result[0] == 0
        assert result[1] == 1
    
    # edge cases test
    def test_edge_cases(self, baseline_jsonl_edge_cases):
        """Test edge cases: interrupted, no numeric, empty"""

        result = parse_baseline_llm_results(baseline_jsonl_edge_cases)
        assert result[0] == 0  # Valid
        assert 1 not in result  # Interrupted - skipped
        assert result[2] == -1  # No numeric - returns -1
        assert result[3] == -1  # Empty text - returns -1


# reasoning llm results 
# parsing tests
class TestParseReasoningLLMResults:

    # valid results test
    def test_valid_results(self, reasoning_jsonl_valid):
        """Test parsing valid reasoning results"""

        result = parse_reasoning_llm_results(reasoning_jsonl_valid)
        assert 0 in result
        assert 1 in result
        assert result[0] == "This is the reasoning text"
        assert result[1] == "Another reasoning"
    
    # edge cases test
    def test_edge_cases(self, reasoning_jsonl_edge_cases):
        """Test edge cases: missing delimiters"""

        result = parse_reasoning_llm_results(reasoning_jsonl_edge_cases)
        assert 0 in result
        assert 1 not in result
        assert 2 in result
        assert result[2] == ""
        assert 3 not in result 


# objective judge results 
# parsing tests
class TestParseObjectiveJudgeResults:

    # valid results test
    def test_valid_results(self, objective_judge_jsonl_valid):
        """Test parsing valid objective judge results"""

        result = parse_objective_judge_results(objective_judge_jsonl_valid)
        assert 0 in result
        assert 1 in result
        assert "faithfulness" in result[0]
        assert "consistency" in result[0]
        assert "coherence" in result[0]
        assert result[0]["faithfulness"] == 4.5
        assert result[0]["consistency"] == 4.0
        assert result[0]["coherence"] == 4.75

    # edge cases test
    def test_edge_cases(self, objective_judge_jsonl_edge_cases):
        """Test edge cases: missing delimiter, invalid JSON"""

        result = parse_objective_judge_results(objective_judge_jsonl_edge_cases)
        assert 0 in result
        assert 1 not in result
        assert 2 in result
        assert 3 not in result
        assert 4 not in result


# zero-shot cot llm results 
# parsing tests
class TestParseZeroShotCotLLMResults:

    # valid results test
    def test_valid_results(self, zero_shot_cot_jsonl_valid):
        """Test parsing valid zero-shot CoT results"""

        result = parse_zero_shot_cot_llm_results(zero_shot_cot_jsonl_valid)
        assert result[0] == 0
        assert result[1] == 1
    
    # edge cases test
    def test_edge_cases(self, zero_shot_cot_jsonl_edge_cases):
        """Test edge cases: interrupted, missing delimiter, no numeric"""

        result = parse_zero_shot_cot_llm_results(zero_shot_cot_jsonl_edge_cases)
        assert result[0] == 0
        assert 1 not in result
        assert result[2] == -1 
        assert result[3] == -1 


# cot llm results 
# parsing tests
class TestParseCotLLMResults:

    # valid results test
    def test_valid_results(self, cot_jsonl_valid):
        """Test parsing valid CoT results with majority vote"""
        result = parse_cot_llm_results(cot_jsonl_valid)
        assert result[0] == 0  # Majority: 2 votes for 0, 1 vote for 1
        assert result[1] == 1  # Single vote for 1
    
    # edge cases test
    def test_edge_cases(self, cot_jsonl_edge_cases):
        """Test edge cases: interrupted, missing delimiter, no numeric, majority vote"""
        result = parse_cot_llm_results(cot_jsonl_edge_cases)

        assert result[0] == -1
        assert result[1] == 1

