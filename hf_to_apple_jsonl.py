#!/usr/bin/env python3
"""
Convert HuggingFace datasets to Apple Foundation Model training format (JSONL).

This script downloads a dataset from HuggingFace and converts it to the JSONL format
expected by Apple's Foundation Model adapter training toolkit.

Expected format: Each line contains a JSON object with conversation turns:
[{"role": "user", "content": "PROMPT"}, {"role": "assistant", "content": "RESPONSE"}]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not found. Install with: pip install datasets")
    sys.exit(1)

# Optional Claude Code SDK integration
CLAUDE_AVAILABLE = False
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    pass


class DatasetAnalyzer:
    """Intelligent dataset structure analyzer with Claude Code SDK integration."""
    
    def __init__(self, use_claude: bool = False):
        self.use_claude = use_claude and CLAUDE_AVAILABLE
        self.analysis_results = {}
        self.conversion_rationale = []
        
    def analyze_dataset_structure(self, dataset, sample_size: int = 10) -> Dict[str, Any]:
        """Analyze dataset structure to determine optimal conversion strategy."""
        sample_data = dataset.select(range(min(sample_size, len(dataset))))
        
        # Basic structure analysis
        field_analysis = self._analyze_fields(sample_data)
        pattern_analysis = self._detect_conversation_patterns(sample_data)
        content_analysis = self._analyze_content_types(sample_data)
        
        analysis = {
            'total_examples': len(dataset),
            'sample_size': len(sample_data),
            'fields': field_analysis,
            'patterns': pattern_analysis,
            'content_types': content_analysis,
            'recommendations': self._generate_recommendations(field_analysis, pattern_analysis)
        }
        
        if self.use_claude:
            analysis = self._enhance_with_claude_analysis(analysis, sample_data)
        
        self.analysis_results = analysis
        return analysis
    
    def _analyze_fields(self, sample_data) -> Dict[str, Any]:
        """Analyze field names and types across samples."""
        field_info = {}
        all_fields = set()
        
        for example in sample_data:
            all_fields.update(example.keys())
        
        for field in all_fields:
            values = [example.get(field) for example in sample_data if field in example]
            non_null_values = [v for v in values if v is not None]
            
            field_info[field] = {
                'presence_rate': len(non_null_values) / len(sample_data),
                'data_types': list(set(type(v).__name__ for v in non_null_values)),
                'sample_values': non_null_values[:3],
                'avg_length': sum(len(str(v)) for v in non_null_values) / len(non_null_values) if non_null_values else 0
            }
        
        return field_info
    
    def _detect_conversation_patterns(self, sample_data) -> Dict[str, Any]:
        """Detect conversation and dialog patterns in the data."""
        patterns = {
            'instruction_response': 0,
            'question_answer': 0,
            'prompt_response': 0,
            'input_output': 0,
            'multi_turn_conversation': 0,
            'single_text_with_markers': 0
        }
        
        for example in sample_data:
            # Check for instruction-response patterns
            if any(field in example for field in ['instruction', 'input', 'prompt']) and \
               any(field in example for field in ['output', 'response', 'answer']):
                if 'instruction' in example:
                    patterns['instruction_response'] += 1
                elif 'input' in example:
                    patterns['input_output'] += 1
                elif 'prompt' in example:
                    patterns['prompt_response'] += 1
            
            # Check for question-answer
            if 'question' in example and 'answer' in example:
                patterns['question_answer'] += 1
            
            # Check for conversation arrays
            for field, value in example.items():
                if isinstance(value, list) and value:
                    if isinstance(value[0], dict) and 'role' in value[0]:
                        patterns['multi_turn_conversation'] += 1
                        break
            
            # Check for text with conversation markers
            for field, value in example.items():
                if isinstance(value, str):
                    if 'Human:' in value and 'Assistant:' in value:
                        patterns['single_text_with_markers'] += 1
                        break
        
        return patterns
    
    def _analyze_content_types(self, sample_data) -> Dict[str, Any]:
        """Analyze content characteristics for better conversion decisions."""
        analysis = {
            'avg_text_lengths': {},
            'language_indicators': [],
            'special_tokens': set(),
            'formatting_patterns': []
        }
        
        for example in sample_data:
            for field, value in example.items():
                if isinstance(value, str):
                    if field not in analysis['avg_text_lengths']:
                        analysis['avg_text_lengths'][field] = []
                    analysis['avg_text_lengths'][field].append(len(value))
                    
                    # Look for special tokens
                    if '<' in value and '>' in value:
                        import re
                        tokens = re.findall(r'<[^>]+>', value)
                        analysis['special_tokens'].update(tokens)
        
        # Calculate averages
        for field in analysis['avg_text_lengths']:
            lengths = analysis['avg_text_lengths'][field]
            analysis['avg_text_lengths'][field] = sum(lengths) / len(lengths) if lengths else 0
        
        analysis['special_tokens'] = list(analysis['special_tokens'])
        return analysis
    
    def _generate_recommendations(self, field_analysis, pattern_analysis) -> List[Dict[str, Any]]:
        """Generate conversion strategy recommendations based on analysis."""
        recommendations = []
        
        # Find the most common pattern
        max_pattern = max(pattern_analysis.items(), key=lambda x: x[1])
        
        if max_pattern[1] > 0:
            confidence = max_pattern[1] / sum(pattern_analysis.values())
            recommendations.append({
                'strategy': max_pattern[0],
                'confidence': confidence,
                'reason': f"Most common pattern detected ({max_pattern[1]} examples)",
                'fields_to_use': self._get_fields_for_strategy(max_pattern[0], field_analysis)
            })
        
        # Add fallback recommendations
        high_presence_fields = {k: v for k, v in field_analysis.items() if v['presence_rate'] > 0.8}
        if high_presence_fields:
            recommendations.append({
                'strategy': 'high_presence_fields',
                'confidence': 0.7,
                'reason': f"Fields with >80% presence rate: {list(high_presence_fields.keys())}",
                'fields_to_use': list(high_presence_fields.keys())
            })
        
        return recommendations
    
    def _get_fields_for_strategy(self, strategy: str, field_analysis: Dict) -> List[str]:
        """Get recommended fields for a specific strategy."""
        strategy_mappings = {
            'instruction_response': ['instruction', 'output'],
            'question_answer': ['question', 'answer'],
            'prompt_response': ['prompt', 'response'],
            'input_output': ['input', 'output'],
        }
        
        base_fields = strategy_mappings.get(strategy, [])
        available_fields = [f for f in base_fields if f in field_analysis and field_analysis[f]['presence_rate'] > 0.5]
        
        return available_fields
    
    def _enhance_with_claude_analysis(self, analysis: Dict, sample_data) -> Dict[str, Any]:
        """Enhance analysis using Claude Code SDK for deeper insights."""
        try:
            client = anthropic.Anthropic()
            
            # Prepare sample for Claude analysis
            sample_str = json.dumps([dict(example) for example in sample_data.select(range(3))], indent=2)
            
            prompt = f"""
            Analyze this dataset sample and provide insights for converting to Apple Foundation Model training format.
            The target format is: [{{"role": "user", "content": "PROMPT"}}, {{"role": "assistant", "content": "RESPONSE"}}]
            
            Dataset sample:
            {sample_str}
            
            Current analysis:
            {json.dumps(analysis, indent=2)}
            
            Please provide:
            1. Best field mapping strategy
            2. Confidence level (0-1)
            3. Potential issues to watch for
            4. Alternative approaches if primary fails
            
            Respond in JSON format with keys: strategy, confidence, issues, alternatives
            """
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            claude_analysis = json.loads(response.content[0].text)
            analysis['claude_insights'] = claude_analysis
            
            self.conversion_rationale.append(f"Claude Analysis: {claude_analysis}")
            
        except Exception as e:
            analysis['claude_insights'] = {'error': f"Claude analysis failed: {str(e)}"}
            self.conversion_rationale.append(f"Claude analysis failed: {str(e)}")
        
        return analysis
    
    def generate_conversion_rationale(self, output_dir: str, analysis: Dict, conversion_stats: Dict):
        """Generate detailed rationale file explaining conversion choices."""
        rationale_path = Path(output_dir) / "conversion_rationale.txt"
        
        with open(rationale_path, 'w', encoding='utf-8') as f:
            f.write("# Dataset Conversion Rationale\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("## Dataset Analysis Summary\n")
            f.write(f"Total examples: {analysis['total_examples']}\n")
            f.write(f"Sample size analyzed: {analysis['sample_size']}\n")
            f.write(f"Successfully converted: {conversion_stats.get('successful', 0)}\n")
            f.write(f"Failed conversions: {conversion_stats.get('failed', 0)}\n")
            f.write(f"Success rate: {conversion_stats.get('success_rate', 0):.2%}\n\n")
            
            f.write("## Field Analysis\n")
            for field, info in analysis['fields'].items():
                f.write(f"- {field}:\n")
                f.write(f"  Presence rate: {info['presence_rate']:.2%}\n")
                f.write(f"  Data types: {info['data_types']}\n")
                f.write(f"  Avg length: {info['avg_length']:.1f} chars\n")
                f.write(f"  Sample values: {info['sample_values']}\n\n")
            
            f.write("## Pattern Detection Results\n")
            for pattern, count in analysis['patterns'].items():
                if count > 0:
                    f.write(f"- {pattern}: {count} examples\n")
            f.write("\n")
            
            f.write("## Conversion Strategy Recommendations\n")
            for i, rec in enumerate(analysis['recommendations'], 1):
                f.write(f"{i}. Strategy: {rec['strategy']}\n")
                f.write(f"   Confidence: {rec['confidence']:.2%}\n")
                f.write(f"   Reason: {rec['reason']}\n")
                f.write(f"   Fields: {rec['fields_to_use']}\n\n")
            
            if 'claude_insights' in analysis:
                f.write("## Claude Code SDK Insights\n")
                f.write(json.dumps(analysis['claude_insights'], indent=2))
                f.write("\n\n")
            
            f.write("## Decision Log\n")
            for entry in self.conversion_rationale:
                f.write(f"- {entry}\n")
            
            f.write("\n## Implementation Notes\n")
            f.write("- Data was not modified during analysis\n")
            f.write("- All conversions preserve original content\n")
            f.write("- Failed conversions are logged but not forced\n")
            f.write("- Format detection uses multiple heuristics for robustness\n")


def _try_intelligent_conversion(example: Dict[str, Any], recommendation: Dict, analyzer: DatasetAnalyzer) -> Optional[List[Dict[str, str]]]:
    """Try conversion using intelligent recommendation."""
    strategy = recommendation['strategy']
    fields = recommendation.get('fields_to_use', [])
    
    if strategy == 'instruction_response' and len(fields) >= 2:
        input_field = next((f for f in fields if f in ['instruction', 'input', 'prompt']), None)
        output_field = next((f for f in fields if f in ['output', 'response', 'answer']), None)
        
        if input_field in example and output_field in example:
            return [
                {"role": "user", "content": str(example[input_field])},
                {"role": "assistant", "content": str(example[output_field])}
            ]
    
    elif strategy == 'question_answer' and 'question' in example and 'answer' in example:
        return [
            {"role": "user", "content": str(example['question'])},
            {"role": "assistant", "content": str(example['answer'])}
        ]
    
    elif strategy == 'multi_turn_conversation':
        # Find conversation field
        for field, value in example.items():
            if isinstance(value, list) and value and isinstance(value[0], dict) and 'role' in value[0]:
                messages = []
                for msg in value:
                    if 'role' in msg and 'content' in msg:
                        messages.append({
                            "role": msg['role'],
                            "content": str(msg['content'])
                        })
                return messages if messages else None
    
    elif strategy == 'single_text_with_markers':
        # Find text field with conversation markers
        for field, value in example.items():
            if isinstance(value, str) and 'Human:' in value and 'Assistant:' in value:
                return _parse_conversation_text(value)
    
    return None


def _parse_conversation_text(text: str) -> Optional[List[Dict[str, str]]]:
    """Parse conversation text with Human:/Assistant: markers."""
    parts = text.split("Human:")
    messages = []
    
    for part in parts[1:]:  # Skip first empty part
        if "Assistant:" in part:
            human_part, assistant_part = part.split("Assistant:", 1)
            messages.extend([
                {"role": "user", "content": human_part.strip()},
                {"role": "assistant", "content": assistant_part.strip()}
            ])
    
    return messages if messages else None


def convert_to_apple_format(example: Dict[str, Any], text_field: str = None, 
                          conversation_field: str = None, analyzer: DatasetAnalyzer = None) -> Optional[List[Dict[str, str]]]:
    """
    Convert a dataset example to Apple's training format with intelligent field detection.
    
    Args:
        example: Single example from the dataset
        text_field: Field name containing the text (for single-turn datasets)
        conversation_field: Field name containing conversation data
        analyzer: DatasetAnalyzer instance for intelligent field mapping
        
    Returns:
        List of message dictionaries or None if conversion fails
    """
    
    # Try intelligent conversion first if analyzer is available
    if analyzer and analyzer.analysis_results:
        recommendations = analyzer.analysis_results.get('recommendations', [])
        
        for rec in recommendations:
            if rec['confidence'] > 0.5:  # Only try high-confidence recommendations
                converted = _try_intelligent_conversion(example, rec, analyzer)
                if converted:
                    analyzer.conversion_rationale.append(
                        f"Successfully used strategy '{rec['strategy']}' with confidence {rec['confidence']:.2%}"
                    )
                    return converted
    
    # Fall back to original conversion logic if intelligent conversion fails
    # Handle conversation-style datasets (like ChatML format)
    if conversation_field and conversation_field in example:
        conversations = example[conversation_field]
        if isinstance(conversations, list):
            messages = []
            for msg in conversations:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    messages.append({
                        "role": msg['role'],
                        "content": str(msg['content'])
                    })
            return messages if messages else None
    
    # Handle instruction-response datasets
    if 'instruction' in example and 'output' in example:
        return [
            {"role": "user", "content": str(example['instruction'])},
            {"role": "assistant", "content": str(example['output'])}
        ]
    
    # Handle input-output datasets
    if 'input' in example and 'output' in example:
        return [
            {"role": "user", "content": str(example['input'])},
            {"role": "assistant", "content": str(example['output'])}
        ]
    
    # Handle question-answer datasets
    if 'question' in example and 'answer' in example:
        return [
            {"role": "user", "content": str(example['question'])},
            {"role": "assistant", "content": str(example['answer'])}
        ]
    
    # Handle prompt-response datasets
    if 'prompt' in example and 'response' in example:
        return [
            {"role": "user", "content": str(example['prompt'])},
            {"role": "assistant", "content": str(example['response'])}
        ]
    
    # Handle text field for single-turn datasets
    if text_field and text_field in example:
        text = str(example[text_field])
        # Simple heuristic: split on common delimiters
        if "Human:" in text and "Assistant:" in text:
            parts = text.split("Human:")
            messages = []
            for part in parts[1:]:  # Skip first empty part
                if "Assistant:" in part:
                    human_part, assistant_part = part.split("Assistant:", 1)
                    messages.extend([
                        {"role": "user", "content": human_part.strip()},
                        {"role": "assistant", "content": assistant_part.strip()}
                    ])
            return messages if messages else None
    
    warning_msg = f"Warning: Could not convert example with keys: {list(example.keys())}"
    print(warning_msg)
    
    if analyzer:
        analyzer.conversion_rationale.append(f"Failed conversion: {warning_msg}")
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace datasets to Apple Foundation Model training format"
    )
    parser.add_argument(
        "dataset_name", 
        help="HuggingFace dataset name (e.g., 'tatsu-lab/alpaca' or 'microsoft/DialoGPT-medium')"
    )
    parser.add_argument(
        "output_dir",
        help="Output directory for the JSONL files"
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--text-field",
        help="Field name containing text data (for single-turn datasets)"
    )
    parser.add_argument(
        "--conversation-field", 
        help="Field name containing conversation data"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Maximum number of examples to process"
    )
    parser.add_argument(
        "--train-split-ratio",
        type=float,
        default=0.9,
        help="Ratio of data to use for training (rest for validation, default: 0.9)"
    )
    parser.add_argument(
        "--use-claude-hook",
        action="store_true",
        help="Enable Claude Code SDK integration for intelligent dataset analysis (requires anthropic package)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset: {args.dataset_name}")
    try:
        dataset = load_dataset(args.dataset_name, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Limit examples if specified
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
        print(f"Limited to {len(dataset)} examples")
    
    # Initialize Claude Code SDK analyzer if requested
    analyzer = None
    if args.use_claude_hook:
        if not CLAUDE_AVAILABLE:
            print("Warning: Claude Code SDK not available. Install with: pip install anthropic")
            print("Proceeding without intelligent analysis...")
        else:
            print("Initializing Claude Code SDK analyzer...")
            analyzer = DatasetAnalyzer(use_claude=True)
            
            print("Analyzing dataset structure...")
            analysis = analyzer.analyze_dataset_structure(dataset, sample_size=min(20, len(dataset)))
            
            print(f"Analysis complete. Found {len(analysis['recommendations'])} conversion strategies.")
            for i, rec in enumerate(analysis['recommendations'], 1):
                print(f"  {i}. {rec['strategy']} (confidence: {rec['confidence']:.1%}) - {rec['reason']}")
    
    # Convert examples
    converted_examples = []
    failed_conversions = 0
    
    print("Converting examples...")
    for i, example in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(dataset)} examples")
        
        converted = convert_to_apple_format(
            example, 
            text_field=args.text_field,
            conversation_field=args.conversation_field,
            analyzer=analyzer
        )
        
        if converted:
            converted_examples.append(converted)
        else:
            failed_conversions += 1
    
    print(f"Successfully converted {len(converted_examples)} examples")
    if failed_conversions > 0:
        print(f"Failed to convert {failed_conversions} examples")
    
    if not converted_examples:
        print("No examples were successfully converted. Check your dataset format.")
        sys.exit(1)
    
    # Split into train/validation
    split_idx = int(len(converted_examples) * args.train_split_ratio)
    train_examples = converted_examples[:split_idx]
    valid_examples = converted_examples[split_idx:]
    
    # Write training file
    train_file = output_path / "train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in train_examples:
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Written {len(train_examples)} training examples to {train_file}")
    
    # Write validation file if we have validation examples
    if valid_examples:
        valid_file = output_path / "valid.jsonl"
        with open(valid_file, 'w', encoding='utf-8') as f:
            for example in valid_examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Written {len(valid_examples)} validation examples to {valid_file}")
    
    # Generate conversion rationale if analyzer was used
    if analyzer:
        conversion_stats = {
            'successful': len(converted_examples),
            'failed': failed_conversions,
            'success_rate': len(converted_examples) / (len(converted_examples) + failed_conversions)
        }
        
        print("Generating conversion rationale...")
        analyzer.generate_conversion_rationale(str(output_path), analyzer.analysis_results, conversion_stats)
        
        rationale_file = output_path / "conversion_rationale.txt"
        print(f"Conversion rationale written to {rationale_file}")
    
    print("Conversion complete!")
    
    # Show first example for verification
    print("\nFirst converted example:")
    print(json.dumps(converted_examples[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()