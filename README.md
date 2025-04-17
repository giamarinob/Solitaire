# Solitaire

# 🃏 Solitaire Agent

This is a Python implementation of Klondike Solitaire, designed to support interaction with a reinforcement learning (RL) agent. The game includes support for human-readable commands, scoring, win/loss conditions, and a timer-based mechanic for evaluating agent performance.

## Features

- Full Klondike Solitaire game logic
- Draw-3 rule from the stock
- Scoring system based on classic Solitaire rules
- Undo functionality (single-step)
- Game timer (5-minute countdown)
- Deterministic win detection
- Command loop for human or AI play
- Designed with RL training in mind

## Scoring Rules

| Action                             | Points   |
|------------------------------------|----------|
| Waste → Tableau                    | +20      |
| Tableau → Foundation               | +100     |
| Waste → Foundation                 | +120     |
| Foundation → Tableau               | −100     |
| Recycle Waste                      | −20      |
| Auto-finish (deterministic win)   | +100 per card in tableau |

## Timer & Bonus

The game includes a 5-minute timer. If the agent finishes early or quits, a time-based bonus is awarded:

Time Bonus = Seconds Remaining × Foundation Multiplier

## Undo

Supports undoing the most recent move only. Game state (cards, score, etc.) is deep-copied to preserve fidelity.

## Getting Started

### Requirements

- Python 3.8+
- No external dependencies

### Run the Game

```bash
python main_manual.py
```

## 🎮 Commands

| Command                       | Description                                            |
|-------------------------------|--------------------------------------------------------|
| `draw`                        | Draw up to 3 cards from the stock to the waste         |
| `move waste to tableau	`      | Move top card from waste to a tableau                  |
| `move waste to foundation`    | Move top card from waste to a foundation               |
| `move tableau to tableau	`    | Move face-up cards from one tableau to another         |
| `move tableau to foundation	` | Move top card from tableau to foundation     |
| `move foundation to tableau`  | Move card from foundation to tableau         |
| `undo`                        | Undo the last move                                     |
| `help`                        | Display the list of available commands |
| `quit`                        | Quit the game                                          |

*Note That <start_card_index> starts at 1*

## TODO
- Implement additional win conditions.
- Add more complex AI behavior.
- Improve user interface (UI) and game interaction.
- Support saving and loading game state.
- Add more detailed documentation for the game and agent.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/giamarinob/Solitaire/blob/main/LICENSE) file for details.
