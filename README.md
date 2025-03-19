# Kinetics Technical Assistant with CrewAI

Enhanced Kinetics Noise Control Technical Assistant application with sophisticated AI agent orchestration using CrewAI.

## Overview

This project enhances the existing Kinetics Technical Assistant by integrating CrewAI to create a multi-agent system that coordinates specialized AI agents to deliver higher quality, more comprehensive answers to technical questions.

Key benefits:

- **Specialized Domain Experts**: Multiple AI agents that focus on different aspects of query processing
- **Advanced Orchestration**: Coordinated workflow that intelligently routes information between agents
- **Improved Sources Integration**: Better integration of knowledge base and web content
- **Self-Improvement Mechanisms**: Performance monitoring, feedback loops, and adaptive learning

## Architecture

![Architecture](docs/images/architecture.png)

The system comprises:

1. **Research Agent**: Specializes in gathering information from knowledge bases and web sources
2. **Document Specialist**: Extracts and processes technical content from documents
3. **Product Expert**: Synthesizes information to deliver comprehensive answers
4. **Orchestration Layer**: Coordinates agent activities and manages workflows
5. **Caching System**: Improves response time for similar queries
6. **Feedback Loop**: Captures performance metrics to improve future responses

## Setup

### Prerequisites

- Python 3.9+
- FastAPI
- Azure services (Storage, Search)
- Required API keys (Anthropic, OpenAI, Grok)
- Qdrant vector database

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/kinetics-technical-assistant-crewai.git
cd kinetics-technical-assistant-crewai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Update the configuration:
```bash
cp config.example.py config.py
# Edit config.py with your API keys and settings
```

4. Run the application:
```bash
python -m uvicorn ta:app --reload
```

## Usage

### Web Interface

Access the web interface at: http://localhost:8000

The interface provides:
- Chat interface for technical queries
- Document upload capabilities
- Analytics dashboard to monitor performance
- Administration tools

### CrewAI Configuration

You can customize the agent workflows in `crew/tasks.py` to optimize for different query types. The system automatically selects the best workflow based on query analysis.

## Configuration Options

Key configuration options in `config.py`:

```python
# CrewAI Settings
ENABLE_CREWAI = True  # Toggle CrewAI integration
ENABLE_AB_TESTING = False  # Enable A/B testing between CrewAI and standard workflow
CREWAI_CACHE_TTL = 24  # Cache time-to-live in hours

# Agent Settings
DEFAULT_RESEARCH_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_DOCUMENT_MODEL = "gpt-4o"
DEFAULT_EXPERT_MODEL = "grok-2-latest"

# API Keys
ANTHROPIC_API_KEY = "your-api-key"
OPENAI_API_KEY = "your-api-key"
GROK_API_KEY = "your-api-key"

# Database Settings
QDRANT_URL = "your-qdrant-url"
QDRANT_API_KEY = "your-qdrant-api-key"
```

## Performance Monitoring

The system includes comprehensive performance monitoring via:

1. CrewAI Performance Dashboard in the Gradio interface
2. Excel logs in `G:/APPS/Technical Assistant/crew_performance.xlsx`
3. Agent feedback loop in `G:/APPS/Technical Assistant/agent_feedback.xlsx`

## Extending the System

### Adding New Agent Types

Create new agent classes in `crew/agents.py`:

```python
from crewai import Agent

new_agent = Agent(
    role="Compliance Specialist",
    goal="Ensure all recommendations comply with industry standards",
    backstory="""You are an expert in acoustic and vibration control regulations,
    with deep knowledge of building codes and industry standards.""",
    tools=[standards_tool],
    verbose=True,
    allow_delegation=True
)
```

### Creating New Workflows

Define new task workflows in `crew/tasks.py`:

```python
def create_compliance_workflow(agents, query):
    research_task = Task(
        description=f"Research compliance requirements for: {query}",
        expected_output="Relevant standards and compliance requirements",
        agent=agents["research_agent"]
    )
    
    compliance_task = Task(
        description="Check if the solution meets all compliance requirements",
        expected_output="Compliance analysis with recommendations",
        agent=agents["compliance_agent"],
        context=[research_task]
    )
    
    return [research_task, compliance_task]
```

## License

MIT