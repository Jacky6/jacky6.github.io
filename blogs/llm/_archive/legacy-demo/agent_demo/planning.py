"""
§04 - Planning 规划模块

Enhanced features:
  1. Task decomposition via LLM
  2. Dependency graph (DAG) with topological sort
  3. Parallel execution of independent subtasks
  4. Dynamic replanning based on feedback
  5. Plan validation and progress tracking
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ──────────────────────────────────────────────
# 1. Data Models
# ──────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # waiting for dependencies


@dataclass
class Task:
    """单个子任务。"""

    id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)  # task ids this depends on
    tool_name: str = ""  # 推荐使用的工具
    result: str = ""
    error: str = ""
    retry_count: int = 0
    max_retries: int = 2

    @property
    def is_ready(self) -> bool:
        """是否可以开始执行（依赖是否满足）。"""
        return self.status in (TaskStatus.PENDING, TaskStatus.BLOCKED)

    @property
    def is_done(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)


# ──────────────────────────────────────────────
# 2. Dependency Graph
# ──────────────────────────────────────────────

class DependencyGraph:
    """
    有向无环图 (DAG) — 任务依赖管理。

    Features:
      - Topological sort for execution order
      - Parallel group detection
      - Cycle detection
      - Dependency resolution
    """

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._edges: dict[str, list[str]] = defaultdict(list)  # task → [dependents]
        self._reverse: dict[str, list[str]] = defaultdict(list)  # task → [dependencies]

    def add_task(self, task: Task):
        self._tasks[task.id] = task
        for dep in task.dependencies:
            self._edges[dep].append(task.id)
            self._reverse[task.id].append(dep)

    def get_ready_tasks(self) -> list[Task]:
        """获取当前可以并行执行的任务。"""
        ready = []
        for task in self._tasks.values():
            if not task.is_ready:
                continue
            # Check if all dependencies are completed
            deps_met = all(
                self._tasks[dep].status == TaskStatus.COMPLETED
                for dep in task.dependencies
                if dep in self._tasks
            )
            if deps_met:
                ready.append(task)
        return ready

    def update_task(self, task_id: str, status: TaskStatus, result: str = "", error: str = ""):
        """更新任务状态。"""
        if task_id not in self._tasks:
            return
        task = self._tasks[task_id]
        task.status = status
        task.result = result
        task.error = error

        # 如果任务失败，标记其依赖项为 blocked
        if status == TaskStatus.FAILED:
            for dependent_id in self._edges.get(task_id, []):
                dep_task = self._tasks.get(dependent_id)
                if dep_task and dep_task.status in (TaskStatus.PENDING, TaskStatus.BLOCKED):
                    dep_task.status = TaskStatus.BLOCKED
                    dep_task.error = f"Dependency {task_id} failed"

        # 如果任务完成，检查是否有之前 blocked 的任务现在可以执行
        if status == TaskStatus.COMPLETED:
            for dependent_id in self._edges.get(task_id, []):
                dep_task = self._tasks.get(dependent_id)
                if dep_task and dep_task.status == TaskStatus.BLOCKED:
                    # 检查是否所有依赖都满足了
                    all_met = all(
                        self._tasks[d].status == TaskStatus.COMPLETED
                        for d in dep_task.dependencies
                        if d in self._tasks
                    )
                    if all_met:
                        dep_task.status = TaskStatus.PENDING

    def topological_sort(self) -> list[str]:
        """拓扑排序——获取执行顺序。"""
        in_degree = defaultdict(int)
        for task_id, task in self._tasks.items():
            if task_id not in in_degree:
                in_degree[task_id] = 0
            for dep in task.dependencies:
                in_degree[task_id] += 1

        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for dependent in self._edges.get(node, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(self._tasks):
            # Cycle detected!
            cycle_nodes = set(self._tasks.keys()) - set(order)
            raise ValueError(f"Dependency cycle detected involving: {cycle_nodes}")

        return order

    def find_parallel_groups(self) -> list[list[str]]:
        """找出可以并行执行的任务组。"""
        try:
            order = self.topological_sort()
        except ValueError:
            # Cycle — return each task as its own group
            return [[tid] for tid in self._tasks]

        groups: list[list[str]] = []
        level = {tid: 0 for tid in order}

        for tid in order:
            for dep in self._tasks[tid].dependencies:
                if dep in level:
                    level[tid] = max(level[tid], level[dep] + 1)

        max_level = max(level.values()) if level else 0
        for l in range(max_level + 1):
            group = [tid for tid, lvl in level.items() if lvl == l]
            if group:
                groups.append(group)

        return groups

    def get_progress(self) -> dict:
        total = len(self._tasks)
        if total == 0:
            return {"total": 0, "completed": 0, "failed": 0, "pending": 0, "progress": 0.0}

        counts = defaultdict(int)
        for task in self._tasks.values():
            counts[task.status.value] += 1

        return {
            "total": total,
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "pending": counts.get("pending", 0) + counts.get("in_progress", 0) + counts.get("blocked", 0),
            "progress": round(counts.get("completed", 0) / total, 2),
        }

    def to_dict(self) -> dict:
        return {
            "tasks": {
                tid: {
                    "id": t.id,
                    "title": t.title,
                    "status": t.status.value,
                    "dependencies": t.dependencies,
                    "result": t.result,
                    "error": t.error,
                }
                for tid, t in self._tasks.items()
            },
            "progress": self.get_progress(),
        }


# ──────────────────────────────────────────────
# 3. Planner (unified entry point)
# ──────────────────────────────────────────────

class Planner:
    """
    增强型规划器 — 向后兼容旧 API。

    Features:
      - LLM-based task decomposition
      - Dependency graph management
      - Parallel execution support
      - Dynamic replanning
    """

    DECOMPOSE_PROMPT = """\
Decompose the following complex task into specific, executable steps.

Task: {task}

For each step, provide:
- A unique ID (step_1, step_2, etc.)
- A clear title
- A brief description
- Any dependencies on other steps (none for first steps)
- Recommended tool to use (if applicable)

Return ONLY a JSON array:
[
    {{
        "id": "step_1",
        "title": "Step title",
        "description": "What to do",
        "dependencies": [],
        "tool_name": "search_web"
    }}
]
"""

    REPLAN_PROMPT = """\
The following plan encountered issues during execution. Create an updated plan.

Original Task: {task}

Completed Steps:
{completed}

Feedback: {feedback}

Generate a revised plan that:
1. Skips already completed steps
2. Addresses the feedback
3. Adds any missing steps

Return ONLY a JSON array of remaining steps (same format as before).
"""

    def __init__(self, llm: Callable, max_replans: int = 3):
        self.llm = llm
        self.max_replans = max_replans
        self._current_graph: Optional[DependencyGraph] = None
        self._replan_count: int = 0

    def decompose_task(self, question: str) -> list[dict]:
        """
        将复杂任务分解为可执行步骤（旧 API 兼容）。
        Returns list of step dicts.
        """
        import json

        prompt = self.DECOMPOSE_PROMPT.format(task=question)
        try:
            result = self.llm.invoke(
                [{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = result.content if hasattr(result, "content") else str(result)
            steps = json.loads(content)
            if isinstance(steps, list):
                return steps
        except Exception:
            pass

        # Fallback: simple 3-step plan
        return [
            {"id": "step_1", "title": "信息收集", "description": "收集相关信息", "dependencies": [], "tool_name": "search_web"},
            {"id": "step_2", "title": "数据分析", "description": "分析收集到的数据", "dependencies": ["step_1"], "tool_name": ""},
            {"id": "step_3", "title": "生成报告", "description": "生成最终报告", "dependencies": ["step_2"], "tool_name": ""},
        ]

    async def adecompose_task(self, question: str, callback=None) -> list[dict]:
        """异步版本（旧 API 兼容）。"""
        return self.decompose_task(question)

    def decompose_to_graph(self, question: str) -> DependencyGraph:
        """
        分解任务并构建依赖图。

        Returns a ready-to-execute DependencyGraph.
        """
        steps = self.decompose_task(question)
        graph = DependencyGraph()

        for step in steps:
            task = Task(
                id=step.get("id", f"step_{len(graph._tasks)}"),
                title=step.get("title", ""),
                description=step.get("description", ""),
                dependencies=step.get("dependencies", []),
                tool_name=step.get("tool_name", ""),
            )
            graph.add_task(task)

        self._current_graph = graph
        return graph

    def replan(self, completed: list[dict], feedback: str) -> list[dict]:
        """
        基于反馈重新规划（旧 API 兼容）。
        """
        import json

        self._replan_count += 1
        if self._replan_count > self.max_replans:
            return completed  # Don't replan too many times

        completed_text = "\n".join(
            f"- {c.get('id', '?')}: {c.get('result', c.get('status', ''))}"
            for c in completed
        )

        prompt = self.REPLAN_PROMPT.format(
            task="当前任务",
            completed=completed_text,
            feedback=feedback,
        )

        try:
            result = self.llm.invoke(
                [{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = result.content if hasattr(result, "content") else str(result)
            steps = json.loads(content)
            if isinstance(steps, list):
                return steps
        except Exception:
            pass

        return completed

    def replan_graph(self, completed_tasks: list[Task], feedback: str) -> DependencyGraph:
        """
        基于反馈重建依赖图。
        """
        completed_dicts = [
            {"id": t.id, "result": t.result, "status": t.status.value}
            for t in completed_tasks
        ]
        new_steps = self.replan(completed_dicts, feedback)

        graph = DependencyGraph()
        for step in new_steps:
            task = Task(
                id=step.get("id", f"step_{len(graph._tasks)}"),
                title=step.get("title", ""),
                description=step.get("description", ""),
                dependencies=step.get("dependencies", []),
                tool_name=step.get("tool_name", ""),
            )
            graph.add_task(task)

        self._current_graph = graph
        return graph

    def get_current_plan(self) -> dict:
        """获取当前计划状态。"""
        if not self._current_graph:
            return {"status": "no_plan", "tasks": []}
        return self._current_graph.to_dict()

    def reset(self):
        """重置规划器。"""
        self._current_graph = None
        self._replan_count = 0
