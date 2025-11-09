"""
Blackjack AI Game
"""

import os
import random
import operator
from typing import TypedDict, Annotated, Sequence, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

#card draw function
def draw_card() -> int:
    # draw a card between 2 and 11 inclusive
    return random.randint(2, 11)

#LLM Setup
LLM_TEMPERATURE = 0.0
_llm = None
if os.getenv("OPENAI_API_KEY"):
    try:
        _llm = ChatOpenAI(model="gpt-5-nano", temperature=LLM_TEMPERATURE)
    except Exception:
        _llm = None

def nl_to_action(user_text: str) -> str:
    """
    Call deaker to make decision to 'HIT' or 'STAND' using the LLM,
    
    """
    # Preferred: one-token classification by prompting for strict output
    if _llm is not None:
        try:
            prompt = (
                "You are a blackjack assistant. Classify the user's intent:\n"
                "- Output exactly one token: HIT or STAND.\n"
                "- HIT means the user wants another card.\n"
                "- STAND means the user does not want another card.\n"
                f"User: {user_text}\n"
            )
            resp = _llm.invoke(prompt).content.strip().upper()
            if resp == "HIT":
                return "HIT"
            if resp == "STAND":
                return "STAND"
            # Soft parse (in case the model adds extra words)
            if "HIT" in resp and "STAND" not in resp:
                return "HIT"
            if "STAND" in resp and "HIT" not in resp:
                return "STAND"
        except Exception:
            pass

    # Fallback: keyword heuristics
    text = user_text.lower()
    hit_words = ["hit", "deal", "another", "one more", "next card", "go again", "draw"]
    stand_words = ["stand", "pass", "no", "stop", "hold", "stick", "i'm good", "im good"]
    if any(w in text for w in hit_words):
        return "HIT"
    if any(w in text for w in stand_words):
        return "STAND"
    # Default conservative fallback
    return "STAND"

#set state
class GameState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    player_scores: dict[str, list[int]]
    player_totals: dict[str, int]
    current_player: str
    game_phase: Literal["setup", "playing", "finished"]
    round_count: int
    dealer_messages: list[str]

#Setup agents
class DealerAgent:
    """Dealer draws cards and announces the winner."""

    def deal_card(self, state: GameState, player_name: str) -> GameState:
        card = draw_card()
        if player_name not in state["player_scores"]:
            state["player_scores"][player_name] = []
        state["player_scores"][player_name].append(card)
        state["player_totals"][player_name] = sum(state["player_scores"][player_name])
        msg = f"Dealer: Dealt {card} to {player_name}. Total: {state['player_totals'][player_name]}"
        state["dealer_messages"].append(msg)
        print(msg)
        return state

    def announce_winner(self, state: GameState) -> str:
        valid = {n: t for n, t in state["player_totals"].items() if t <= 21}
        if not valid:
            return "No winner this round."
        name, total = max(valid.items(), key=lambda x: x[1])
        return f"Winner: {name} with {total} points!"

class RandomAIPlayer:
    """AI that randomly decides to hit or stand ."""

    def __init__(self, name: str):
        self.name = name

    def decide_action(self, current_total: int, cards_drawn: int) -> bool:
        # Can't draw beyond 3 cards or when already 21+
        if cards_drawn >= 3 or current_total >= 21:
            return False
        return random.choice([True, False])

#Init Game
def initialize_game(state: GameState) -> GameState:
    print("\n" + "*" * 60)
    print("BLACKJACK AI GAME")
    print("*" * 60)

    state["player_scores"] = {}
    state["player_totals"] = {}
    state["game_phase"] = "setup"
    state["round_count"] = 0
    state["dealer_messages"] = []
    return state

def setup_players(state: GameState) -> GameState:
    print("\n Setting up players...")
    # Three random AIs
    for name in ["AI_Player_1", "AI_Player_2", "AI_Player_3"]:
        state["player_scores"][name] = []
        state["player_totals"][name] = 0
    # Human
    state["player_scores"]["You"] = []
    state["player_totals"]["You"] = 0

    state["game_phase"] = "playing"
    state["current_player"] = "You"
    print("Players list : You, AI_Player_1, AI_Player_2, AI_Player_3")
    return state

def _play_human_turn(state: GameState, dealer: DealerAgent) -> GameState:
    # If no cards yet, deal first one automatically to start
    if len(state["player_scores"]["You"]) == 0:
        state = dealer.deal_card(state, "You")

    while True:
        cards = state["player_scores"]["You"]
        total = state["player_totals"]["You"]
        print(f"\n Your current total: {total}")
        print(f" Your cards: {cards}")

        # Hard stops
        if len(cards) >= 3:
            print("You've drawn 3 cards (maximum). Your turn ends.")
            break
        if total >= 21:
            print("Blackjack!" if total == 21 else " :x Over 21!")
            break

        # Natural language input, interpreted by LLM (with fallback)
        user_text = input("Say what you want (e.g., 'deal me one', 'I'll pass'): ").strip()
        intent = nl_to_action(user_text)  # 'HIT' or 'STAND'

        if intent == "HIT":
            state = dealer.deal_card(state, "You")
            # loop continues to allow up to 3 cards or until bust/stand
        else:
            print("You stand.")
            break

    return state

def _play_ai_turn(state: GameState, dealer: DealerAgent, name: str) -> GameState:
    # If no cards yet, deal first one automatically to start
    if len(state["player_scores"][name]) == 0:
        state = dealer.deal_card(state, name)

    ai = RandomAIPlayer(name)
    while True:
        cards = state["player_scores"][name]
        total = state["player_totals"][name]

        if len(cards) >= 3:
            print(f"\n {name} reached 3 cards. Turn ends.")
            break
        if total >= 21:
            print(f"\n{name}: {'Blackjack! ' if total == 21 else 'Busted! '} (Total: {total})")
            break

        if ai.decide_action(total, len(cards)):
            print(f"\n {name}: Hit (random).")
            state = dealer.deal_card(state, name)
        else:
            print(f"\n {name}: Stand (random).")
            break

    return state

def play_turn(state: GameState) -> GameState:
    dealer = DealerAgent()
    player = state["current_player"]

    if player == "You":
        state = _play_human_turn(state, dealer)
    else:
        state = _play_ai_turn(state, dealer, player)

    return state

def next_player(state: GameState) -> GameState:
    players = ["You", "AI_Player_1", "AI_Player_2", "AI_Player_3"]
    idx = players.index(state["current_player"])
    if idx < len(players) - 1:
        state["current_player"] = players[idx + 1]
        state["round_count"] += 1
        print("\n" + "-" * 60)
    else:
        state["game_phase"] = "finished"
    return state

def end_game(state: GameState) -> GameState:
    print("\n" + "-" * 60)
    print("GAME OVER | FINAL RESULTS")
    print("-" * 60)
    for player, cards in state["player_scores"].items():
        total = state["player_totals"][player]
        status = "✓" if total <= 21 else "✗ BUST"
        print(f"{player}: {cards} = {total} {status}")

    dealer = DealerAgent()
    print(f"\n{dealer.announce_winner(state)}")
    print( "-" * 60)
    return state

#Build Graph
def build_game_graph():
    workflow = StateGraph(GameState)
    workflow.add_node("initialize", initialize_game)
    workflow.add_node("setup", setup_players)
    workflow.add_node("play", play_turn)
    workflow.add_node("next", next_player)
    workflow.add_node("end", end_game)

    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "setup")
    # Each play() handles a full player turn (looping internally),
    # then we move to next player or end.
    workflow.add_edge("setup", "play")
    workflow.add_edge("play", "next")
    workflow.add_conditional_edges(
        "next",
        lambda s: "play" if s["game_phase"] == "playing" else "end",
        {"play": "play", "end": "end"},
    )
    workflow.add_edge("end", END)
    return workflow.compile()

#main function
def main():
    game = build_game_graph()
    initial_state = {
        "messages": [],
        "player_scores": {},
        "player_totals": {},
        "current_player": "",
        "game_phase": "setup",
        "round_count": 0,
        "dealer_messages": [],
    }
    game.invoke(initial_state)
    print("\n Thanks for playing! ")

if __name__ == "__main__":
    main()
