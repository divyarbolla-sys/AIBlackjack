# Blackjack Multi-Agent Game with LangGraph

A command-line Blackjack game featuring 3 AI players built with LangGraph.

## Features

- **Multi-Agent System**: 3 AI players 
- **AI Dealer Agent**: Manages card distribution and game flow
- **Natural Interaction**: Players request cards from the dealer naturally
- **LangGraph State Management**: Uses state graphs for game flow

## Requirements

- Python 3.12+
- OpenAI API key (for AI decision-making and intent classification)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-openai-api-key
```

## Usage

Run the game:
```bash
python main.py
```

### Game Rules

1. Each player (you + 3 AI agents) can draw up to **3 cards**
2. Players must **ask the dealer** to draw cards (cannot draw directly)
3. Goal: Get the **highest score under 21**
4. If you go over 21, you **bust** and lose
5. The winner has the highest score ≤ 21

## Architecture

### State Management
- Uses LangGraph's `StateGraph` for game flow
- State includes: player scores, current player, game phase, messages

### Agents

1. **Dealer Agent**
   - Manages card distribution
   - Announces winners
   - Controls game flow


### Game Flow

```mermaid
Initialize → Setup Players → Play Turn → Next Player → End Game
                                ↓↑ (loop while player wants cards)
```


