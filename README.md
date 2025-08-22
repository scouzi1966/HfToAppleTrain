# HuggingFace to Apple Training Format Converter

Convert HuggingFace datasets to Apple Foundation Model training format (JSONL) with intelligent format detection and Claude Code SDK integration.

## Features

- **Intelligent Dataset Analysis**: Automatically detects conversation patterns and optimal field mappings
- **Claude Code SDK Integration**: Optional AI-powered analysis for complex dataset structures
- **Multiple Format Support**: Handles instruction-response, question-answer, multi-turn conversations, and more
- **Comprehensive Logging**: Generates detailed conversion rationale with decision explanations
- **Backward Compatible**: Works with or without Claude Code SDK

## Installation

```bash
# Basic installation
pip install -r requirements.txt

# For Claude Code SDK integration (optional)
pip install anthropic
```

## Usage

### Basic Usage
```bash
python hf_to_apple_jsonl.py dataset_name output_dir
```

### With Claude Code SDK Integration
```bash
python hf_to_apple_jsonl.py dataset_name output_dir --use-claude-hook
```

### Advanced Options
```bash
python hf_to_apple_jsonl.py dataset_name output_dir \
    --split train \
    --max-examples 1000 \
    --train-split-ratio 0.9 \
    --text-field "text" \
    --conversation-field "messages" \
    --use-claude-hook
```

## Output Format

The script converts datasets to Apple's expected format:
```json
[
  {"role": "user", "content": "PROMPT"},
  {"role": "assistant", "content": "RESPONSE"}
]
```

## Supported Dataset Patterns

- **Instruction-Response**: `instruction` + `output` fields
- **Question-Answer**: `question` + `answer` fields  
- **Prompt-Response**: `prompt` + `response` fields
- **Input-Output**: `input` + `output` fields
- **Multi-turn Conversations**: Arrays with `role` and `content` fields
- **Text with Markers**: Single text fields with `Human:` and `Assistant:` markers

## Output Files

- `train.jsonl`: Training examples
- `valid.jsonl`: Validation examples (if split ratio < 1.0)
- `conversion_rationale.txt`: Detailed analysis and decision log (with `--use-claude-hook`)

## Examples

```bash
# Convert Alpaca dataset
python hf_to_apple_jsonl.py tatsu-lab/alpaca ./output

# Convert with intelligent analysis
python hf_to_apple_jsonl.py microsoft/DialoGPT-medium ./output --use-claude-hook

# Convert specific field mapping
python hf_to_apple_jsonl.py dataset_name ./output --text-field "conversation"
```

## Requirements

- Python 3.7+
- datasets >= 2.0.0
- huggingface_hub >= 0.15.0
- anthropic >= 0.25.0 (optional, for Claude Code SDK)

## License

MIT License