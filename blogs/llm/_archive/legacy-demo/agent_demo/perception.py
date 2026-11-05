"""
Perception Layer — Agent 的"眼睛和耳朵"

Responsibilities:
  - Multi-modal input processing (text / image / audio / file / web)
  - Intent classification (question / command / analysis / creation / chitchat)
  - Complexity & urgency estimation
  - Sentiment analysis
  - Routing suggestion for downstream modules

Design:
  - PerceptionRouter selects the right module per input_type
  - TextPerception handles text: intent + entities + complexity + urgency + sentiment
  - ImagePerception handles images: vision description + optional OCR
  - PerceptionResult is a Pydantic model for structured output
  - Compatible with the demo's existing AgentState TypedDict
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. Structured Result
# ──────────────────────────────────────────────

class Intent(str, Enum):
    QUESTION = "question"
    COMMAND = "command"
    ANALYSIS = "analysis"
    CREATION = "creation"
    CHITCHAT = "chitchat"


class Complexity(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class PerceptionResult(BaseModel):
    """Structured perception output — feeds into AgentState['perception']"""

    intent: Intent = Field(description="User intent classification")
    understood: str = Field(description="Concise summary of what user wants")
    key_entities: list[str] = Field(
        default_factory=list, description="Important names, dates, numbers"
    )
    language: str = Field(default="zh", description="Detected language code")
    complexity: Complexity = Field(description="Input complexity level")
    urgency: Urgency = Field(description="Urgency level")
    sentiment: Sentiment = Field(description="Emotional tone")
    requires_tool: bool = Field(description="Whether a tool call is likely needed")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Confidence score")
    raw_description: str = Field(
        default="", description="Full description (especially for image/audio input)"
    )
    needs_followup: bool = Field(
        default=False, description="Whether more perception is needed"
    )

    # -- Routing helper --
    def route_to(self) -> str:
        """Map intent to downstream module name (matches graph.py node names)."""
        if self.intent == Intent.CHITCHAT:
            return "chitchat"
        if self.intent in (Intent.COMMAND, Intent.ANALYSIS, Intent.CREATION):
            return "tool_use"
        # question → needs reasoning, may or may not need tools
        return "reasoning" if not self.requires_tool else "tool_use"


# ──────────────────────────────────────────────
# 2. Unified Input / Output
# ──────────────────────────────────────────────

@dataclass
class PerceptionInput:
    """Unified perception input wrapper."""

    content: Any  # raw content (str text, image URL, file path, etc.)
    input_type: str = "text"  # text | image | audio | file | web
    source: Optional[str] = None  # URL, filename, etc.
    metadata: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """Hash for caching perception results."""
        raw = json.dumps(
            {"content": str(self.content), "type": self.input_type},
            ensure_ascii=False,
            default=str,
        )
        return hashlib.md5(raw.encode()).hexdigest()


# ──────────────────────────────────────────────
# 3. Intent Classifier (lightweight LLM call)
# ──────────────────────────────────────────────

INTENT_CLASSIFIER_PROMPT = """\
Analyze the following user input and return a JSON object matching this schema:

{{
    "intent": "question|command|analysis|creation|chitchat",
    "understood": "one-sentence summary of what the user wants",
    "key_entities": ["list of important names, dates, numbers, places"],
    "language": "detected language code (e.g., zh, en, ja)",
    "complexity": "simple|medium|complex",
    "urgency": "low|medium|high",
    "sentiment": "positive|negative|neutral",
    "requires_tool": true|false,
    "confidence": 0.0-1.0
}}

Classification guide:
- question: asking for information or opinion
- command: requesting a specific action to be performed
- analysis: requesting data analysis, comparison, or evaluation
- creation: requesting content generation (code, text, report, etc.)
- chitchat: casual conversation, greetings, no specific task

Input: {content}
"""


class IntentClassifier:
    """
    Lightweight intent classifier using LLM structured output.

    Uses the demo's existing LLM instance. In production, you could
    swap this for a cheaper/faster model (e.g., gpt-4o-mini).
    """

    def __init__(self, llm):
        self.llm = llm

    def classify(self, content: str) -> PerceptionResult:
        """Classify text input and return structured PerceptionResult."""
        prompt = INTENT_CLASSIFIER_PROMPT.format(content=content[:2000])

        try:
            wrapper = self.llm.with_structured_output(PerceptionResult)
            raw = wrapper.invoke(prompt)
            # Handle both dict and model return types
            if isinstance(raw, PerceptionResult):
                return raw
            # Dict fallback — construct PerceptionResult
            return PerceptionResult(
                intent=raw.get("intent", "question"),
                understood=raw.get("understood", content[:200]),
                key_entities=raw.get("key_entities", []),
                language=raw.get("language", "zh"),
                complexity=raw.get("complexity", "medium"),
                urgency=raw.get("urgency", "medium"),
                needs_tool=raw.get("needs_tool", False),
            )
        except Exception as e:
            logger.warning("Intent classification failed: %s — using fallback", e)
            return self._fallback(content)

    def _fallback(self, content: str) -> PerceptionResult:
        """Rule-based fallback when LLM is unavailable."""
        lower = content.lower()
        # Heuristic intent detection
        if any(w in lower for w in ["你好", "hello", "hi ", "hey", "在吗", "在嘛"]):
            intent = Intent.CHITCHAT
        elif any(w in lower for w in ["分析", "对比", "评估", "evaluate", "compare"]):
            intent = Intent.ANALYSIS
        elif any(w in lower for w in ["写", "生成", "create", "build", "make"]):
            intent = Intent.CREATION
        elif any(w in lower for w in ["帮我", "请", "do ", "run ", "execute"]):
            intent = Intent.COMMAND
        else:
            intent = Intent.QUESTION

        return PerceptionResult(
            intent=intent,
            understood=f"[fallback] {content[:100]}",
            key_entities=[],
            language="zh" if any("\u4e00" <= c <= "\u9fff" for c in content) else "en",
            complexity=Complexity.MEDIUM,
            urgency=Urgency.MEDIUM,
            sentiment=Sentiment.NEUTRAL,
            requires_tool=intent != Intent.CHITCHAT,
            confidence=0.5,
        )


# ──────────────────────────────────────────────
# 4. Text Perception Module
# ──────────────────────────────────────────────

class TextPerception:
    """
    Text perception: intent classification + entity extraction + complexity/urgency.
    Delegates to IntentClassifier.
    """

    def __init__(self, llm):
        self.classifier = IntentClassifier(llm)

    def supports(self, input_type: str) -> bool:
        return input_type == "text"

    def process(self, inp: PerceptionInput) -> PerceptionResult:
        content = str(inp.content)
        result = self.classifier.classify(content)
        result.raw_description = content[:500]
        return result


# ──────────────────────────────────────────────
# 5. Image Perception Module
# ──────────────────────────────────────────────

class ImagePerception:
    """
    Image perception: visual description via Vision-Language Model.

    Requires an LLM that supports multi-modal input (e.g., Qwen-VL, GPT-4V).
    The demo uses Qwen models via DashScope; Qwen-VL is available.
    """

    VISION_PROMPT_TEXT = (
        "Describe this image in detail. What do you see? "
        "What information does it convey? "
        "If there is text, charts, or data, describe them."
    )

    def __init__(self, vision_llm):
        self.vision_llm = vision_llm

    def supports(self, input_type: str) -> bool:
        return input_type in ("image", "screenshot")

    def process(self, inp: PerceptionInput) -> PerceptionResult:
        """
        Process image input.

        For LangChain-compatible multi-modal LLMs, send the image URL/path
        as an image_url content block.
        """
        try:
            content_parts = [
                {"type": "text", "text": self.VISION_PROMPT_TEXT},
            ]
            # Support both URL and local path
            img_content = inp.content
            if isinstance(img_content, str) and (
                img_content.startswith("http") or img_content.startswith("/")
            ):
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": img_content}}
                )
            else:
                # For base64 or other formats, pass as text description
                content_parts.append({"type": "text", "text": f"[Image content: {img_content}]"})

            msg = [{"role": "user", "content": content_parts}]
            vision_result = self.vision_llm.invoke(msg)
            description = vision_result.content

            # Now classify the description for intent/entities
            classifier = IntentClassifier(self.vision_llm)
            classification = classifier.classify(
                f"[Image description]: {description}\n\nUser query: {inp.metadata.get('query', '')}"
            )
            classification.raw_description = description
            return classification

        except Exception as e:
            logger.warning("Image perception failed: %s — using fallback", e)
            return PerceptionResult(
                intent=Intent.QUESTION,
                understood=f"[Image fallback] {str(inp.content)[:200]}",
                key_entities=[],
                complexity=Complexity.MEDIUM,
                urgency=Urgency.MEDIUM,
                sentiment=Sentiment.NEUTRAL,
                requires_tool=False,
                confidence=0.3,
                needs_followup=True,
            )


# ──────────────────────────────────────────────
# 6. Perception Router
# ──────────────────────────────────────────────

class PerceptionRouter:
    """
    Routes PerceptionInput to the appropriate module based on input_type.

    Usage:
        router = PerceptionRouter(llm=llm, vision_llm=vision_llm)
        result = router.process(PerceptionInput(content="...", input_type="text"))
        print(result.intent, result.route_to())
    """

    def __init__(self, llm, vision_llm=None):
        self.modules: list = []
        self.register(TextPerception(llm))
        if vision_llm:
            self.register(ImagePerception(vision_llm))

    def register(self, module):
        self.modules.append(module)

    def process(self, inp: PerceptionInput) -> PerceptionResult:
        """Route to the right module and process."""
        for module in self.modules:
            if module.supports(inp.input_type):
                return module.process(inp)

        # Fallback: treat as text
        logger.warning(
            "No module supports input_type='%s' — falling back to text perception",
            inp.input_type,
        )
        text_module = TextPerception(self.modules[0].classifier.llm if self.modules else None)
        return text_module.process(PerceptionInput(content=str(inp.content), input_type="text"))


# ──────────────────────────────────────────────
# 7. Perception Cache (optional)
# ──────────────────────────────────────────────

class PerceptionCache:
    """
    Simple in-memory cache for perception results keyed by content hash.
    TTL-based expiration.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self._cache: dict[str, tuple[float, PerceptionResult]] = {}
        self._ttl = ttl_seconds

    def get(self, inp: PerceptionInput) -> Optional[PerceptionResult]:
        import time

        key = inp.content_hash()
        if key in self._cache:
            ts, result = self._cache[key]
            if time.time() - ts < self._ttl:
                return result
            del self._cache[key]
        return None

    def put(self, inp: PerceptionInput, result: PerceptionResult):
        import time

        self._cache[inp.content_hash()] = (time.time(), result)


# ──────────────────────────────────────────────
# 8. Convenience: perceive() one-liner
# ──────────────────────────────────────────────

def perceive(text: str, llm) -> PerceptionResult:
    """
    Quick perception of a text input.

    Usage:
        result = perceive("帮我分析一下这张数据表", llm)
        print(result.intent)  # Intent.ANALYSIS
        print(result.route_to())  # "tool_use"
    """
    classifier = IntentClassifier(llm)
    return classifier.classify(text)
