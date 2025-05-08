from src.Solitaire import Solitaire

def print_help():
    print("""
Available commands:
  ** Note Start Card Index starts at 1 **
  move tableau <from_index> tableau <to_index> <start_card_index>
  move tableau <from_index> foundation
  move waste tableau <to_index>
  move waste foundation
  move foundation <from_index> tableau <to_index>
  draw
  undo
  show
  help
  quit
""")

def main():
    game = Solitaire()
    print("Welcome to Solitaire!\n")
    print_help()
    print("\nInitial Game State:\n")
    game.display()

    while True:
        if game.is_won():
            print("Congratulations! You've won the game!")
            break

        if not game.time_remaining():
            print("Game Over!")
            game.end()
            break

        command = input("\n> ").strip().lower()
        if command == "quit":
            game.end()
            game.display()
            break
        elif command == "help":
            print_help()
        elif command == "show":
            game.display()
        elif command == "draw":
            game.draw_from_stock()
            print("Drew cards from stock.")
        elif command == "undo":
            if game.undo():
                print("Move Reverted")
            else:
                print("No Move to Undo")
        elif command.startswith("move "):
            parts = command.split()
            try:
                if parts[1] == "tableau" and parts[3] == "tableau":
                    game.move("tableau", int(parts[2]), "tableau", int(parts[4]), int(parts[5]))
                elif parts[1] == "tableau" and parts[3] == "foundation":
                    game.move("tableau", int(parts[2]), "foundation", int(parts[4]))
                elif parts[1] == "waste" and parts[2] == "tableau":
                    game.move("waste", None, "tableau", int(parts[3]))
                elif parts[1] == "waste" and parts[2] == "foundation":
                    game.move("waste", None, "foundation", int(parts[3]))
                elif parts[1] == "foundation" and parts[3] == "tableau":
                    game.move("foundation", int(parts[2]), "tableau", int(parts[4]))
                else:
                    print("Invalid move command.")
            except (IndexError, ValueError) as e:
                print(f"{e}")
                print("Invalid command format.")
        else:
            print("Unknown command. Type 'help' to see available commands.")

        game.display()

if __name__ == "__main__":
    main()
