"""
Exercise 2: Memory Systems for Agents
Build and compare different memory approaches for intelligent agents

OBJECTIVE: Understand how memory affects agent behavior and performance
DIFFICULTY: Intermediate
TIME: 25-30 minutes
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Memory:
    """Base memory unit"""
    content: str
    timestamp: float
    importance: float
    memory_type: str
    metadata: Dict[str, Any] = None


class BufferMemory:
    """Simple buffer memory - remembers recent items"""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.memories: List[Memory] = []

    def store(self, content: str, importance: float = 1.0, memory_type: str = "general"):
        """Store a memory"""
        memory = Memory(
            content=content,
            timestamp=time.time(),
            importance=importance,
            memory_type=memory_type,
            metadata={"index": len(self.memories)}
        )

        self.memories.append(memory)

        # Remove oldest if over capacity
        if len(self.memories) > self.max_size:
            removed = self.memories.pop(0)
            print(f"🗑️ Removed old memory: {removed.content[:50]}...")

    def retrieve_recent(self, count: int = 5) -> List[Memory]:
        """Get most recent memories"""
        return self.memories[-count:]

    def retrieve_all(self) -> List[Memory]:
        """Get all memories"""
        return self.memories.copy()

    def clear(self):
        """Clear all memories"""
        self.memories.clear()
        print("🧹 Buffer memory cleared")


class SummarizingMemory:
    """Memory that summarizes old information to save space"""

    def __init__(self, max_detailed: int = 5, summary_threshold: int = 10):
        self.max_detailed = max_detailed
        self.summary_threshold = summary_threshold
        self.detailed_memories: List[Memory] = []
        self.summaries: List[str] = []

    def store(self, content: str, importance: float = 1.0, memory_type: str = "general"):
        """Store a memory with automatic summarization"""
        memory = Memory(
            content=content,
            timestamp=time.time(),
            importance=importance,
            memory_type=memory_type
        )

        self.detailed_memories.append(memory)

        # Trigger summarization if needed
        if len(self.detailed_memories) > self.summary_threshold:
            self._create_summary()

    def _create_summary(self):
        """Create summary of oldest memories"""
        # Take oldest memories for summarization
        to_summarize = self.detailed_memories[:self.max_detailed]
        self.detailed_memories = self.detailed_memories[self.max_detailed:]

        # Simple summarization (in practice, you'd use an LLM)
        summary_content = f"Summary of {len(to_summarize)} memories: "
        key_points = [mem.content[:30] + "..." for mem in to_summarize]
        summary_content += "; ".join(key_points)

        self.summaries.append(summary_content)
        print(f"📝 Created summary: {summary_content}")

    def retrieve_all(self) -> Dict[str, Any]:
        """Get all memories and summaries"""
        return {
            "detailed_memories": self.detailed_memories,
            "summaries": self.summaries
        }

    def clear(self):
        """Clear all memories"""
        self.detailed_memories.clear()
        self.summaries.clear()
        print("🧹 Summarizing memory cleared")


class ImportanceBasedMemory:
    """Memory that prioritizes important information"""

    def __init__(self, max_size: int = 15):
        self.max_size = max_size
        self.memories: List[Memory] = []

    def store(self, content: str, importance: float = 1.0, memory_type: str = "general"):
        """Store memory with importance-based retention"""
        memory = Memory(
            content=content,
            timestamp=time.time(),
            importance=importance,
            memory_type=memory_type
        )

        self.memories.append(memory)

        # Remove least important if over capacity
        if len(self.memories) > self.max_size:
            # Sort by importance (ascending) and remove least important
            self.memories.sort(key=lambda m: m.importance)
            removed = self.memories.pop(0)
            print(f"🗑️ Removed low-importance memory: {removed.content[:50]}... (importance: {removed.importance})")

    def retrieve_by_importance(self, min_importance: float = 0.5) -> List[Memory]:
        """Retrieve memories above importance threshold"""
        return [mem for mem in self.memories if mem.importance >= min_importance]

    def retrieve_top_k(self, k: int = 5) -> List[Memory]:
        """Get top k most important memories"""
        sorted_memories = sorted(self.memories, key=lambda m: m.importance, reverse=True)
        return sorted_memories[:k]

    def clear(self):
        """Clear all memories"""
        self.memories.clear()
        print("🧹 Importance-based memory cleared")


class SemanticMemory:
    """Memory that organizes information by semantic similarity"""

    def __init__(self):
        self.memories: Dict[str, List[Memory]] = {}
        self.categories = ["facts", "procedures", "experiences", "goals"]

    def store(self, content: str, category: str = "general", importance: float = 1.0):
        """Store memory in semantic categories"""
        if category not in self.memories:
            self.memories[category] = []

        memory = Memory(
            content=content,
            timestamp=time.time(),
            importance=importance,
            memory_type=category
        )

        self.memories[category].append(memory)
        print(f"🗂️ Stored in category '{category}': {content[:50]}...")

    def retrieve_by_category(self, category: str) -> List[Memory]:
        """Retrieve memories from specific category"""
        return self.memories.get(category, [])

    def search_across_categories(self, query_keywords: List[str]) -> List[Memory]:
        """Search for memories containing keywords"""
        results = []
        query_lower = [kw.lower() for kw in query_keywords]

        for category, memories in self.memories.items():
            for memory in memories:
                content_lower = memory.content.lower()
                if any(kw in content_lower for kw in query_lower):
                    results.append(memory)

        return results

    def get_categories(self) -> Dict[str, int]:
        """Get categories and their memory counts"""
        return {cat: len(mems) for cat, mems in self.memories.items()}

    def clear(self):
        """Clear all memories"""
        self.memories.clear()
        print("🧹 Semantic memory cleared")


def compare_memory_systems():
    """Exercise: Compare different memory systems"""

    print("🧠 Memory Systems Comparison Exercise")
    print("="*40)

    # Initialize different memory systems
    buffer_mem = BufferMemory(max_size=5)
    summarizing_mem = SummarizingMemory(max_detailed=3, summary_threshold=5)
    importance_mem = ImportanceBasedMemory(max_size=5)
    semantic_mem = SemanticMemory()

    # Test data - conversation with varying importance
    conversations = [
        {"content": "User said hello and introduced themselves as John", "importance": 0.8, "category": "facts"},
        {"content": "Discussed the weather - sunny today", "importance": 0.2, "category": "experiences"},
        {"content": "User asked about Python programming best practices", "importance": 0.9, "category": "goals"},
        {"content": "Mentioned they work at TechCorp as a developer", "importance": 0.7, "category": "facts"},
        {"content": "Asked about lunch recommendations", "importance": 0.3, "category": "experiences"},
        {"content": "Wants to learn about machine learning", "importance": 1.0, "category": "goals"},
        {"content": "Shared that they have 5 years programming experience", "importance": 0.8, "category": "facts"},
        {"content": "Complained about traffic this morning", "importance": 0.1, "category": "experiences"},
        {"content": "Needs help with a specific ML project deadline next week", "importance": 0.95, "category": "goals"},
        {"content": "Mentioned they prefer coffee over tea", "importance": 0.2, "category": "facts"},
    ]

    print("\\n📝 Storing conversations in all memory systems...")

    for i, conv in enumerate(conversations):
        print(f"\\nStoring: {conv['content']}")

        # Store in all systems
        buffer_mem.store(conv['content'], conv['importance'])
        summarizing_mem.store(conv['content'], conv['importance'])
        importance_mem.store(conv['content'], conv['importance'])
        semantic_mem.store(conv['content'], conv['category'], conv['importance'])

    print("\\n" + "="*60)
    print("MEMORY SYSTEM COMPARISON RESULTS")
    print("="*60)

    # Compare retrieval
    print("\\n🔄 BUFFER MEMORY (Recent 3):")
    recent = buffer_mem.retrieve_recent(3)
    for mem in recent:
        print(f"   • {mem.content}")

    print("\\n📋 SUMMARIZING MEMORY:")
    summarizing_data = summarizing_mem.retrieve_all()
    print(f"   Detailed memories: {len(summarizing_data['detailed_memories'])}")
    print(f"   Summaries: {len(summarizing_data['summaries'])}")
    for summary in summarizing_data['summaries']:
        print(f"   📝 {summary}")

    print("\\n⭐ IMPORTANCE-BASED MEMORY (High importance):")
    important = importance_mem.retrieve_by_importance(0.7)
    for mem in important:
        print(f"   • {mem.content} (importance: {mem.importance})")

    print("\\n🗂️ SEMANTIC MEMORY (By category):")
    for category, count in semantic_mem.get_categories().items():
        print(f"   {category}: {count} memories")
        if category == "goals":  # Show goals as example
            goals = semantic_mem.retrieve_by_category("goals")
            for goal in goals:
                print(f"     • {goal.content}")

    return {
        "buffer": buffer_mem,
        "summarizing": summarizing_mem,
        "importance": importance_mem,
        "semantic": semantic_mem
    }


def memory_retrieval_exercise(memory_systems):
    """Exercise: Practice memory retrieval scenarios"""

    print("\\n🎯 Memory Retrieval Scenarios Exercise")
    print("="*42)

    scenarios = [
        {
            "query": "What does the user want to learn?",
            "best_system": "semantic",
            "category": "goals"
        },
        {
            "query": "What was just discussed recently?",
            "best_system": "buffer",
            "category": None
        },
        {
            "query": "What are the most critical things to remember?",
            "best_system": "importance",
            "category": None
        },
        {
            "query": "What do we know about the user's background?",
            "best_system": "semantic",
            "category": "facts"
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\\n📋 Scenario {i}: {scenario['query']}")
        print("Which memory system would work best?")
        print("a) Buffer Memory (recent items)")
        print("b) Summarizing Memory (with summaries)")
        print("c) Importance-Based Memory")
        print("d) Semantic Memory (categorized)")

        user_choice = input("Your choice (a/b/c/d): ").lower().strip()
        choice_map = {"a": "buffer", "b": "summarizing", "c": "importance", "d": "semantic"}

        if user_choice in choice_map:
            chosen_system = choice_map[user_choice]

            if chosen_system == scenario["best_system"]:
                print("✅ Excellent choice!")
            else:
                print(f"🔄 Consider: {scenario['best_system']} memory might be more suitable")

            # Demonstrate retrieval
            print(f"\\n🔍 Retrieving from {chosen_system} memory:")

            if chosen_system == "buffer":
                results = memory_systems["buffer"].retrieve_recent(3)
                for mem in results:
                    print(f"   • {mem.content}")

            elif chosen_system == "importance":
                results = memory_systems["importance"].retrieve_top_k(3)
                for mem in results:
                    print(f"   • {mem.content} (importance: {mem.importance})")

            elif chosen_system == "semantic" and scenario["category"]:
                results = memory_systems["semantic"].retrieve_by_category(scenario["category"])
                for mem in results:
                    print(f"   • {mem.content}")

            elif chosen_system == "summarizing":
                data = memory_systems["summarizing"].retrieve_all()
                print(f"   Recent details: {len(data['detailed_memories'])} memories")
                print(f"   Summaries: {len(data['summaries'])}")

        else:
            print("❌ Invalid choice")

        print("-" * 50)


def design_custom_memory():
    """Exercise: Design a custom memory system"""

    print("\\n🛠️ Custom Memory System Design Exercise")
    print("="*45)

    print("Design a memory system for a personal AI assistant that:")
    print("• Remembers user preferences")
    print("• Learns from interactions")
    print("• Prioritizes recent and important information")
    print("• Can forget outdated information")

    design_questions = [
        "What types of information should be stored?",
        "How would you determine importance?",
        "What triggers should cause forgetting?",
        "How would you organize the memories?",
        "How would retrieval work?",
        "How would the system learn and adapt?"
    ]

    user_design = {}
    print("\\nAnswer the following design questions:")

    for i, question in enumerate(design_questions, 1):
        print(f"\\n{i}. {question}")
        answer = input("Your answer: ").strip()
        user_design[f"question_{i}"] = answer

    print("\\n🎨 YOUR CUSTOM MEMORY SYSTEM DESIGN:")
    print("-" * 40)
    for i, (key, answer) in enumerate(user_design.items(), 1):
        print(f"{i}. {design_questions[i-1]}")
        print(f"   Answer: {answer}")

    # Provide expert feedback
    print("\\n🤖 EXPERT DESIGN CONSIDERATIONS:")
    print("-" * 35)
    expert_considerations = [
        "Use hybrid approach: buffer + importance + semantic categories",
        "Importance = recency + user engagement + explicit ratings",
        "Forget based on: time decay + low importance + user request",
        "Organize by: topics, relationships, user goals, temporal clusters",
        "Retrieval: semantic search + importance ranking + recency bias",
        "Learning: user feedback, behavior patterns, success metrics"
    ]

    for i, consideration in enumerate(expert_considerations, 1):
        print(f"{i}. {consideration}")

    print("\\n💡 Implementation Challenge:")
    print("Try implementing one aspect of your design as a Python class!")


def main():
    """Run all memory exercises"""

    print("🧠 MEMORY SYSTEMS EXERCISES")
    print("="*30)

    print("\\nThese exercises explore different approaches to agent memory:")
    print("1. Compare Memory Systems")
    print("2. Memory Retrieval Practice")
    print("3. Design Custom Memory")

    # # Run comparison exercise
    # memory_systems = compare_memory_systems()

    while True:
        print("\\n" + "="*40)
        choice = input("\\nChoose exercise (1-3), 'next' for next exercise, or 'q' to quit: ").strip().lower()

        if choice == '1':
            memory_systems = compare_memory_systems()
        elif choice == '2':
            memory_retrieval_exercise(memory_systems)
        elif choice == '3':
            design_custom_memory()
        elif choice in ['next', 'n']:
            print("\\n➡️ Continue to exercise_3_tool_integration.py")
            break
        elif choice in ['q', 'quit']:
            print("\\n👋 Great work on memory systems! Continue when ready.")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, 'next', or 'q'")


if __name__ == "__main__":
    main()