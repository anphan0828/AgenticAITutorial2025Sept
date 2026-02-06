"""
API Comparison Exercise
Compare OpenAI and Anthropic APIs for different tasks

OBJECTIVE: Understand the strengths and differences between major LLM APIs
DIFFICULTY: Intermediate
TIME: 30-40 minutes
"""

import os
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
from dotenv import load_dotenv

# Import API clients
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

load_dotenv()


@dataclass
class APIResponse:
    """Standardized response format"""
    model: str
    provider: str
    response_text: str
    response_time: float
    token_count: Optional[int]
    success: bool
    error: Optional[str] = None


class APIComparator:
    """Compare responses from different API providers"""

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None

        # Initialize OpenAI client
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize Anthropic client
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def call_openai(self, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 500) -> APIResponse:
        """Call OpenAI API"""
        if not self.openai_client:
            return APIResponse(
                model=model,
                provider="openai",
                response_text="",
                response_time=0,
                token_count=None,
                success=False,
                error="OpenAI client not available"
            )

        start_time = time.time()

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )

            response_time = time.time() - start_time
            response_text = response.choices[0].message.content

            return APIResponse(
                model=model,
                provider="openai",
                response_text=response_text,
                response_time=response_time,
                token_count=response.usage.total_tokens if response.usage else None,
                success=True
            )

        except Exception as e:
            return APIResponse(
                model=model,
                provider="openai",
                response_text="",
                response_time=time.time() - start_time,
                token_count=None,
                success=False,
                error=str(e)
            )

    def call_anthropic(self, prompt: str, model: str = "claude-3-5-haiku-latest", max_tokens: int = 500) -> APIResponse:
        """Call Anthropic API"""
        if not self.anthropic_client:
            return APIResponse(
                model=model,
                provider="anthropic",
                response_text="",
                response_time=0,
                token_count=None,
                success=False,
                error="Anthropic client not available"
            )

        start_time = time.time()

        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            response_time = time.time() - start_time
            response_text = ""

            # Extract text from response
            for content_block in response.content:
                if hasattr(content_block, 'text'):
                    response_text += content_block.text

            return APIResponse(
                model=model,
                provider="anthropic",
                response_text=response_text,
                response_time=response_time,
                token_count=response.usage.output_tokens if hasattr(response, 'usage') else None,
                success=True
            )

        except Exception as e:
            return APIResponse(
                model=model,
                provider="anthropic",
                response_text="",
                response_time=time.time() - start_time,
                token_count=None,
                success=False,
                error=str(e)
            )

    def compare_responses(self, prompt: str, task_type: str = "general") -> Dict[str, APIResponse]:
        """Compare responses from both providers"""
        print(f"\\n🔍 Comparing responses for: {task_type.upper()}")
        print(f"Prompt: {prompt}")
        print("-" * 60)

        results = {}

        # Test OpenAI
        if self.openai_client:
            print("🧪 Testing OpenAI...")
            openai_response = self.call_openai(prompt)
            results["openai"] = openai_response

            if openai_response.success:
                print(f"   ✅ Success in {openai_response.response_time:.2f}s")
            else:
                print(f"   ❌ Error: {openai_response.error}")
        else:
            print("⚠️  OpenAI not available")

        # Test Anthropic
        if self.anthropic_client:
            print("🧪 Testing Anthropic...")
            anthropic_response = self.call_anthropic(prompt)
            results["anthropic"] = anthropic_response

            if anthropic_response.success:
                print(f"   ✅ Success in {anthropic_response.response_time:.2f}s")
            else:
                print(f"   ❌ Error: {anthropic_response.error}")
        else:
            print("⚠️  Anthropic not available")

        return results

    def print_detailed_comparison(self, results: Dict[str, APIResponse]):
        """Print detailed comparison of responses"""
        print("\\n" + "="*80)
        print("DETAILED RESPONSE COMPARISON")
        print("="*80)

        for provider, response in results.items():
            if not response.success:
                print(f"\\n❌ {provider.upper()}: Failed - {response.error}")
                continue

            print(f"\\n🤖 {provider.upper()} ({response.model})")
            print("-" * 50)
            print(f"⏱️  Response Time: {response.response_time:.2f} seconds")
            print(f"📊 Token Count: {response.token_count if response.token_count else 'N/A'}")
            print(f"📝 Response Length: {len(response.response_text)} characters")
            print(f"💬 Response:\\n{response.response_text}\\n")

        # Comparison metrics
        successful_responses = {k: v for k, v in results.items() if v.success}

        if len(successful_responses) >= 2:
            print("📈 COMPARISON METRICS")
            print("-" * 25)

            # Speed comparison
            fastest = min(successful_responses.keys(),
                         key=lambda x: successful_responses[x].response_time)
            print(f"🚀 Fastest: {fastest} ({successful_responses[fastest].response_time:.2f}s)")

            # Length comparison
            most_detailed = max(successful_responses.keys(),
                               key=lambda x: len(successful_responses[x].response_text))
            print(f"📝 Most detailed: {most_detailed} ({len(successful_responses[most_detailed].response_text)} chars)")

            # Token efficiency (if available)
            responses_with_tokens = {k: v for k, v in successful_responses.items() if v.token_count}
            if responses_with_tokens:
                most_efficient = min(responses_with_tokens.keys(),
                                   key=lambda x: responses_with_tokens[x].token_count)
                print(f"💡 Most token-efficient: {most_efficient} ({responses_with_tokens[most_efficient].token_count} tokens)")


def reasoning_task_comparison():
    """Compare APIs on reasoning tasks"""
    print("🧠 REASONING TASK COMPARISON")
    print("="*35)

    comparator = APIComparator()

    reasoning_prompts = [
        {
            "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Think step by step.",
            "task_type": "logical_reasoning",
            "expected_answer": "$0.05"
        },
        {
            "prompt": "If all roses are flowers and all flowers are plants, and some plants are trees, can we conclude that some roses are trees? Explain your reasoning.",
            "task_type": "deductive_reasoning",
            "expected_answer": "No, we cannot conclude that"
        },
        {
            "prompt": "You have 8 balls that look identical. 7 of them weigh the same, but 1 is slightly heavier. You have a balance scale. What's the minimum number of weighings needed to find the heavy ball?",
            "task_type": "optimization",
            "expected_answer": "2 weighings"
        }
    ]

    for i, test_case in enumerate(reasoning_prompts, 1):
        print(f"\\n{'='*60}")
        print(f"REASONING TEST {i}: {test_case['task_type'].upper()}")
        print(f"{'='*60}")

        results = comparator.compare_responses(test_case["prompt"], test_case["task_type"])
        comparator.print_detailed_comparison(results)

        print(f"\\n💡 Expected answer: {test_case['expected_answer']}")

        # User evaluation
        print("\\n🎯 EVALUATION:")
        for provider, response in results.items():
            if response.success:
                correctness = input(f"Rate {provider}'s correctness (1-5): ").strip()
                clarity = input(f"Rate {provider}'s clarity (1-5): ").strip()
                print(f"   {provider}: Correctness={correctness}, Clarity={clarity}")

        input("\\nPress Enter to continue to next test...")


def creative_task_comparison():
    """Compare APIs on creative tasks"""
    print("\\n🎨 CREATIVE TASK COMPARISON")
    print("="*30)

    comparator = APIComparator()

    creative_prompts = [
        {
            "prompt": "Write a short story (2-3 paragraphs) about a robot who discovers emotions. Make it touching and memorable.",
            "task_type": "creative_writing"
        },
        {
            "prompt": "Create a marketing slogan for a new eco-friendly water bottle. Make it catchy and memorable.",
            "task_type": "marketing_copy"
        },
        {
            "prompt": "Brainstorm 5 innovative features for a future smartphone that don't exist yet. Be creative but practical.",
            "task_type": "brainstorming"
        }
    ]

    for i, test_case in enumerate(creative_prompts, 1):
        print(f"\\n{'='*60}")
        print(f"CREATIVE TEST {i}: {test_case['task_type'].upper()}")
        print(f"{'='*60}")

        results = comparator.compare_responses(test_case["prompt"], test_case["task_type"])
        comparator.print_detailed_comparison(results)

        # User evaluation
        print("\\n🎯 EVALUATION:")
        for provider, response in results.items():
            if response.success:
                creativity = input(f"Rate {provider}'s creativity (1-5): ").strip()
                engagement = input(f"Rate {provider}'s engagement (1-5): ").strip()
                print(f"   {provider}: Creativity={creativity}, Engagement={engagement}")

        input("\\nPress Enter to continue to next test...")


def technical_task_comparison():
    """Compare APIs on technical tasks"""
    print("\\n💻 TECHNICAL TASK COMPARISON")
    print("="*32)

    comparator = APIComparator()

    technical_prompts = [
        {
            "prompt": "Explain how to implement a binary search algorithm. Include code in Python and explain the time complexity.",
            "task_type": "algorithm_explanation"
        },
        {
            "prompt": "Debug this Python code and explain what's wrong:\\n\\ndef find_max(arr):\\n    max_val = 0\\n    for num in arr:\\n        if num > max_val:\\n            max_val = num\\n    return max_val",
            "task_type": "code_debugging"
        },
        {
            "prompt": "Design a simple REST API for a todo list application. Include endpoints, HTTP methods, and expected request/response formats.",
            "task_type": "system_design"
        }
    ]

    for i, test_case in enumerate(technical_prompts, 1):
        print(f"\\n{'='*60}")
        print(f"TECHNICAL TEST {i}: {test_case['task_type'].upper()}")
        print(f"{'='*60}")

        results = comparator.compare_responses(test_case["prompt"], test_case["task_type"])
        comparator.print_detailed_comparison(results)

        # User evaluation
        print("\\n🎯 EVALUATION:")
        for provider, response in results.items():
            if response.success:
                accuracy = input(f"Rate {provider}'s technical accuracy (1-5): ").strip()
                completeness = input(f"Rate {provider}'s completeness (1-5): ").strip()
                print(f"   {provider}: Accuracy={accuracy}, Completeness={completeness}")

        input("\\nPress Enter to continue to next test...")


def performance_benchmark():
    """Benchmark API performance"""
    print("\\n⚡ PERFORMANCE BENCHMARK")
    print("="*25)

    comparator = APIComparator()

    # Simple prompt for speed testing
    benchmark_prompt = "Explain the concept of machine learning in exactly 100 words."

    print("Running speed benchmark (5 requests each)...")

    results = {"openai": [], "anthropic": []}

    # Test each provider 5 times
    for i in range(5):
        print(f"\\nRound {i + 1}/5:")

        if comparator.openai_client:
            openai_response = comparator.call_openai(benchmark_prompt)
            if openai_response.success:
                results["openai"].append(openai_response.response_time)
                print(f"   OpenAI: {openai_response.response_time:.2f}s")

        if comparator.anthropic_client:
            anthropic_response = comparator.call_anthropic(benchmark_prompt)
            if anthropic_response.success:
                results["anthropic"].append(anthropic_response.response_time)
                print(f"   Anthropic: {anthropic_response.response_time:.2f}s")

        time.sleep(1)  # Rate limiting

    # Calculate statistics
    print("\\n📊 PERFORMANCE RESULTS:")
    print("-" * 25)

    for provider, times in results.items():
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"{provider.upper()}:")
            print(f"   Average: {avg_time:.2f}s")
            print(f"   Fastest: {min_time:.2f}s")
            print(f"   Slowest: {max_time:.2f}s")
            print(f"   Consistency: {(1 - (max_time - min_time) / avg_time) * 100:.1f}%")
        else:
            print(f"{provider.upper()}: No successful requests")


def main():
    """Run API comparison exercises"""

    print("🥊 API COMPARISON EXERCISES")
    print("="*30)

    # Check API availability
    print("🔍 Checking API Availability:")
    print(f"   OpenAI: {'✅ Available' if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY') else '❌ Not available'}")
    print(f"   Anthropic: {'✅ Available' if ANTHROPIC_AVAILABLE and os.getenv('ANTHROPIC_API_KEY') else '❌ Not available'}")

    if not ((OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY')) or
            (ANTHROPIC_AVAILABLE and os.getenv('ANTHROPIC_API_KEY'))):
        print("\\n❌ No APIs available. Please:")
        print("1. Install required packages: pip install openai anthropic")
        print("2. Set API keys in .env file")
        return

    print("\\nThis exercise compares OpenAI and Anthropic APIs across different task types.")
    print("You'll evaluate responses for quality, accuracy, and creativity.")

    exercises = [
        ("Reasoning Tasks", reasoning_task_comparison),
        ("Creative Tasks", creative_task_comparison),
        ("Technical Tasks", technical_task_comparison),
        ("Performance Benchmark", performance_benchmark)
    ]

    while True:
        print("\\n" + "="*40)
        print("Available exercises:")
        for i, (name, _) in enumerate(exercises, 1):
            print(f"{i}. {name}")
        print("q. Quit")

        choice = input("\\nChoose exercise (1-4) or 'q' to quit: ").strip().lower()

        if choice == 'q':
            break
        elif choice in ['1', '2', '3', '4']:
            idx = int(choice) - 1
            name, exercise_func = exercises[idx]

            print(f"\\n{'='*50}")
            print(f"STARTING: {name.upper()}")
            print(f"{'='*50}")

            try:
                exercise_func()
            except KeyboardInterrupt:
                print("\\n⏸️ Exercise interrupted")
            except Exception as e:
                print(f"❌ Exercise error: {e}")

            print(f"\\n✅ {name} completed!")

        else:
            print("❌ Invalid choice")

    print("\\n🎉 API Comparison exercises completed!")
    print("\\n💡 Key Insights:")
    print("1. Different models excel at different types of tasks")
    print("2. Response time varies by provider and model complexity")
    print("3. Quality evaluation often depends on the specific use case")
    print("4. Both APIs have strengths in reasoning, creativity, and technical tasks")
    print("5. Performance benchmarks help in choosing the right API for your needs")


if __name__ == "__main__":
    main()