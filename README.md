# Chess Environment

| Attribute             | Description |
|-----------------------|-------------|
| **Import**            | `import chess_environment.chess_environment_v0` |
| **Actions**           | Discrete |
| **Parallel API**      | No |
| **Manual Control**    | No |
| **Agents**            | `agents = ['white', 'black']` |
| **Number of Agents**  | 2 |
| **Action Shape**      | `Discrete(4130)` |
| **Action Values**     | `Discrete(4130)` |
| **Observation Shape** | `{ "board": (8, 8), "kingMoved": (2,), "rookMoved": (4,), "enPassantLocation": (2,), "actionMask": (4130,) }` |
| **Observation Values**| `{ "board": [0-12], "kingMoved": [0-1], "rookMoved": [0-1], "enPassantLocation": [0-7], "actionMask": [0-1] }` |

## Observation Space
The observation is a dictionary which contains an 'observation' element described below and an 'action_mask' which holds all the legal moves that can be made, with the index representing the move, 0 representing and invalid move, and a 1 representing a valid move.

### Observation
Observation is a dictionary which contains 'board', 'kingMoved', 'rookMoved', and 'enPassantLocation'

- Board - Contains an 8x8 grid with values 0-12 that represent different pieces.
    - 0: Empty Square
    - 1: Black Rook
    - 2: Black Knight
    - 3: Black Bishop
    - 4: Black Queen
    - 5: Black King
    - 6: Black Pawn
    - 7: White Rook
    - 8: White Knight
    - 9: White Bishop
    - 10: White Queen
    - 11: White King
    - 12: White Pawn
- kingMoved - Dictionary containing keys 'white' and 'black' with values 0 for False and 1 for True in regards to whether or not they have been moved before.
- rookMoved - Dictionary containing keys 'whiteKingside', 'whiteQueenside', 'blackKingside', and 'blackQueenside' with values 0 for False and 1 for True in regards to whether or not they have been moved before.
- enPassantLocation - List containing 2 values, both being 0 if there is no valid en passant. If there is a valid pawn to en passant the first value will be the row it moved to and the second value will be the column it moved to.

## Action Space
Actions 0-4095 are normal moves/captures that can be turned into start and end positions of a piece by:
```Python
startRow, startCol = divmod(action//64, 8)
endRow, endCol = divmod(action&64, 8)
```
Actions 4096-4127 are promotion moves that can be found by:
```Python
endCol = (action - 4096) // 4
promotionPiece = (action - 4096) % 4 # 0 = Queen, 1 = Rook, 2 = Bishop, 3 = Knight
startRow = 1 if agent == "white" else 6
startCol = endCol
endRow = 7 if startRow == 6 else 0
```
Actions 4128 and 4129 are castling moves which can be found by:
```Python
if action == 4128:
    kingRow = 7 if agent == "white" else 0
    return [kingRow, 4], [kingRow, 6], -1  # Kingside castling
if action == 4129:
    kingRow = 7 if agent == "white" else 0
    return [kingRow, 4], [kingRow, 2], -2  # Queenside castling
```

## Rewards
Winner: +1
Loser: -1
Draw: 0