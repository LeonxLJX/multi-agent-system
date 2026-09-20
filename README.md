# 🤝 Multi-Agent System — Specialized Agents Collaborate

**A team of AI agents with different roles work together to solve complex tasks.** One plans, one researches, one writes — they communicate, debate, and deliver a final result.

---

## 🚀 What It Does

You say: *"Write a market analysis for AI coding tools"*

The system breaks it down:
1. 📋 **Planner Agent** → "Research market size → List competitors → Write report"
2. 🔍 **Researcher Agent** → "Market size is $2.3B, competitors are GitHub Copilot, Cursor..."
3. ✍️ **Writer Agent** → Synthesizes research into a polished report
4. ✅ **Reviewer Agent** → Checks accuracy, flags gaps

This is how real AI products are built today — not one giant prompt, but **specialized agents collaborating**.

---

## 🧠 Why Multi-Agent?

A single LLM doing everything = bad at everything. Specialization wins:
- 🎯 **Focus**: Each agent has one job, prompt is short and precise
- 🔄 **Iteration**: Agents can critique and refine each other's work
- 🧪 **Testability**: Each agent can be tested independently
- 📈 **Scalability**: Add more agents, not bigger prompts

---

## ⚙️ Architecture

```
👤 User: "Write a market analysis for AI coding tools"
         ↓
📋 Planner Agent ──→ task breakdown
         ↓
    ┌────┴────┐
    ↓         ↓
🔍 Researcher  ✍️ Writer
    ↓         ↓
    └────┬────┘
         ↓
    ✅ Reviewer Agent ──→ final report
```

---

## 📁 Structure

```
multi-agent-system/
├── src/
│   ├── agents/         # each agent = one file
│   │   ├── planner.py
│   │   ├── researcher.py
│   │   ├── writer.py
│   │   └── reviewer.py
│   ├── orchestrator.py # coordinates all agents
│   └── app.py          # FastAPI entry
├── examples/
│   └── market_analysis.py  # example task
└── README.md
```

---

## 🎯 Agent Roles

| Agent | Role | What it does |
|-------|------|-------------|
| 📋 Planner | Break down the task | Decompose complex goal into steps |
| 🔍 Researcher | Gather information | Search, fetch, summarize |
| ✍️ Writer | Produce the deliverable | Write report/code/document |
| ✅ Reviewer | Quality check | Verify accuracy, flag issues |

---

## 🚀 Run

```bash
pip install openai fastapi uvicorn
export OPENAI_API_KEY=sk-xxx
python examples/market_analysis.py
```

---

**Built with ❤️ using Python + OpenAI Multi-Agent Pattern**
