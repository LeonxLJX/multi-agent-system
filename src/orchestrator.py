"""
Multi-Agent Orchestrator — coordinates specialized agents.
"""
from openai import OpenAI

client = OpenAI()

class BaseAgent:
    """Base class for all agents."""
    def __init__(self, role: str, goal: str, model: str = "gpt-4o-mini"):
        self.role = role
        self.goal = goal
        self.model = model

    def run(self, task: str, context: str = "") -> str:
        """Execute the task with given context."""
        prompt = f"""
        Role: {self.role}
        Goal: {self.goal}
        
        Context from previous agents:
        {context}
        
        Task: {task}
        
        Respond concisely and professionally.
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class PlannerAgent(BaseAgent):
    """Breaks down complex tasks into steps."""
    def __init__(self):
        super().__init__(
            role="Task Planner",
            goal="Break down complex tasks into clear, actionable steps.",
        )


class ResearcherAgent(BaseAgent):
    """Gathers and summarizes information."""
    def __init__(self):
        super().__init__(
            role="Research Analyst",
            goal="Gather relevant facts and data, summarize concisely.",
        )


class WriterAgent(BaseAgent):
    """Produces polished deliverables."""
    def __init__(self):
        super().__init__(
            role="Professional Writer",
            goal="Produce clear, well-structured deliverables.",
        )


class ReviewerAgent(BaseAgent):
    """Reviews work for accuracy and completeness."""
    def __init__(self):
        super().__init__(
            role="Quality Reviewer",
            goal="Check work for accuracy, consistency, and completeness.",
        )


class Orchestrator:
    """Coordinates all agents to solve a task."""

    def __init__(self):
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()

    def solve(self, user_task: str) -> dict:
        """Run the full multi-agent pipeline."""
        print(f"📋 Planner: breaking down '{user_task}'...")
        steps = self.planner.run(user_task)

        print(f"🔍 Researcher: gathering info...")
        research = self.researcher.run(user_task, steps)

        print(f"✍️ Writer: producing deliverable...")
        draft = self.writer.run(user_task, f"{steps}\n\n{research}")

        print(f"✅ Reviewer: checking quality...")
        review = self.reviewer.run(draft, f"Original task: {user_task}")

        return {
            "task": user_task,
            "plan": steps,
            "research": research,
            "draft": draft,
            "review": review,
        }
