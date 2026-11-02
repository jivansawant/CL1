# Practical 5: Reinforcement Learning — Tic-Tac-Toe in Python

Since you're new to reinforcement learning, let's start from **zero**. First, understand what reinforcement learning means, then we'll understand the game, and finally we'll go through the code line by line.

You do not need to know reinforcement learning before starting this practical.

## 1. CORE TAKEAWAY

**Reinforcement Learning (RL) is a type of machine learning in which an agent learns what to do by interacting with an environment and receiving rewards or penalties.**

In this practical, we'll train a computer to play Tic-Tac-Toe. It will learn which moves are useful by playing games, receiving rewards for winning, penalties for losing, and learning from its experience.

## 2. UNDERSTAND REINFORCEMENT LEARNING FIRST

### What is machine learning?

Machine learning is a way of teaching computers to learn patterns from data or experience instead of manually programming every decision.

There are three commonly taught types:

| Type | How it learns | Simple example |
|---|---|---|
| Supervised learning | Learns from examples with correct answers | Predicting whether an email is spam |
| Unsupervised learning | Finds patterns in data without provided answers | Grouping similar flowers |
| Reinforcement learning | Learns by taking actions and receiving rewards or penalties | Learning to play Tic-Tac-Toe |

In the previous practical, K-Means was unsupervised learning. Here, we'll use reinforcement learning.

### What exactly is reinforcement learning?

Imagine teaching a child to play Tic-Tac-Toe.

1. The child makes a move.
2. The opponent responds.
3. The game continues.
4. If the child wins, the child receives a reward.
5. If the child loses, the child receives a penalty.
6. Over many games, the child learns which moves tend to lead to winning.

Reinforcement learning works on a similar principle. Instead of a child, we have a computer program called an **agent**.

The agent learns from the consequences of its actions.

### Five important terms you must understand

Let's use our Tic-Tac-Toe game to understand each term.

**1. Agent**

The agent is the learner or decision-maker.

In our practical, the agent is the computer program learning to play Tic-Tac-Toe.

**2. Environment**

The environment is the world in which the agent operates.

Here, the environment is the Tic-Tac-Toe game: the board, the rules, the opponent, and the conditions for winning or losing.

**3. State**

A state describes the current situation.

For example, suppose the board looks like this:

```text
 X | O | X
---+---+---
   | X |  
---+---+---
 O |   | O
```

This arrangement of Xs, Os, and empty cells is the current state.

The agent examines the state to decide its next move.

**4. Action**

An action is something the agent chooses to do.

In Tic-Tac-Toe, an action means placing X in one of the empty cells.

For example, the agent might choose the bottom-middle cell.

**5. Reward**

A reward is feedback that tells the agent how good or bad an outcome was.

For our practical, we can use these rewards:

| Game result | Reward |
|---|---:|
| Agent wins | +1 |
| Agent loses | -1 |
| Draw | 0 |

These numbers are design choices. We use them to encourage the agent to win and discourage it from losing.

### How does the agent actually learn?

This is the most important part.

Suppose the agent plays 1,000 games.

Initially, it may make poor moves because it has little experience. After each game, it updates what it has learned about the moves it took.

Over time, it can learn that certain moves tend to lead to victories, while others tend to lead to losses.

This is called **learning through trial and error**.

However, the agent does not automatically become intelligent just because it plays many games. Its learning depends on the algorithm, reward system, exploration, and quality of its training experience.

## 3. MINIMALIST PYTHON CODE

We'll build the practical in five parts, matching your assignment:

1. Setting up the environment
2. Defining the Tic-Tac-Toe game
3. Building the reinforcement learning model
4. Training the model
5. Testing the model

To keep the code understandable, our agent will play as **X**, and the opponent will play as **O** using random moves.

Our learning algorithm will be **Q-learning**, a common reinforcement learning method.

### Complete Python code

Run the following in one Jupyter Notebook cell.

```python
import random
import numpy as np
from collections import defaultdict

# A. Setting up the environment
q_table = defaultdict(lambda: np.zeros(9))
alpha = 0.1
gamma = 0.9
epsilon = 1.0

# B. Defining the Tic-Tac-Toe game
def check_winner(board, mark):
    wins = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]

    return any(all(board[i] == mark for i in line) for line in wins)


def available_moves(board):
    return [i for i in range(9) if board[i] == " "]


def show_board(board):
    for i in range(0, 9, 3):
        print(" | ".join(board[i:i+3]))
        if i < 6:
            print("--+---+--")


# C. Building the reinforcement learning model
def choose_action(state, moves, explore=True):
    if explore and random.random() < epsilon:
        return random.choice(moves)

    values = q_table[state]
    best_value = max(values[i] for i in moves)
    best_moves = [i for i in moves if values[i] == best_value]
    return random.choice(best_moves)


def update_q(state, action, reward, next_state=None, next_moves=None):
    old_value = q_table[state][action]

    if next_state is None:
        future_value = 0
    else:
        future_value = max(q_table[next_state][i] for i in next_moves)

    q_table[state][action] = old_value + alpha * (
        reward + gamma * future_value - old_value
    )


# D. Training the model
for episode in range(20000):
    board = [" "] * 9

    while True:
        state = tuple(board)
        moves = available_moves(board)

        # Agent plays X
        action = choose_action(state, moves)
        board[action] = "X"

        if check_winner(board, "X"):
            update_q(state, action, 1)
            break

        if not available_moves(board):
            update_q(state, action, 0)
            break

        # Random opponent plays O
        board[random.choice(available_moves(board))] = "O"

        if check_winner(board, "O"):
            update_q(state, action, -1)
            break

        if not available_moves(board):
            update_q(state, action, 0)
            break

        next_state = tuple(board)
        next_moves = available_moves(board)
        update_q(state, action, 0, next_state, next_moves)

    epsilon = max(0.05, epsilon * 0.9998)

print("Training completed!")


# E. Testing the trained model
wins = 0
losses = 0
draws = 0

for game in range(1000):
    board = [" "] * 9

    while True:
        state = tuple(board)
        moves = available_moves(board)

        # Trained agent plays X
        action = choose_action(state, moves, explore=False)
        board[action] = "X"

        if check_winner(board, "X"):
            wins += 1
            break

        if not available_moves(board):
            draws += 1
            break

        # Random opponent plays O
        board[random.choice(available_moves(board))] = "O"

        if check_winner(board, "O"):
            losses += 1
            break

        if not available_moves(board):
            draws += 1
            break

print("Results after 1000 test games:")
print("Wins:", wins)
print("Losses:", losses)
print("Draws:", draws)
print("Win percentage:", round(wins / 1000 * 100, 2), "%")
```

**Important:** The results can vary between runs because the opponent's moves and some action choices are random. The training rewards are designed for the agent to win, avoid losses, and learn through experience.

---

## 4. CODE WALKTHROUGH — UNDERSTAND EVERY PART

We'll go slowly. You don't need to memorize every line immediately. First understand what each section does and why.

# Part A: Setting up the environment

```python
import random
import numpy as np
from collections import defaultdict
```

### What do these imports mean?

**1. `import random`**

This lets Python make random choices.

For example:

```python
random.choice([0, 1, 2])
```

It randomly chooses one of the numbers.

In our practical, the opponent uses random moves during training and testing.

**2. `import numpy as np`**

NumPy helps us store and work with numbers efficiently.

Our model needs to store nine values for each board state, one for each possible cell on the board.

**3. `from collections import defaultdict`**

This creates a dictionary that automatically supplies a default value when a new key is accessed.

We'll use it to store the agent's learned knowledge.

Now look at the next lines:

```python
q_table = defaultdict(lambda: np.zeros(9))
alpha = 0.1
gamma = 0.9
epsilon = 1.0
```

These four lines are central to the reinforcement learning model.

### What is a Q-table?

Imagine the agent sees this board:

```text
   |   |  
---+---+---
   | X |  
---+---+---
 O |   |  
```

The agent needs to decide which empty cell to select.

It stores a value for each possible move. These values represent how useful the agent currently estimates each action to be in that state.

That collection of learned values is called the **Q-table**.

In our code:

```python
q_table = defaultdict(lambda: np.zeros(9))
```

- The key is the current board state.
- The value is an array of nine numbers.
- Each number represents the estimated value of choosing the corresponding cell.

For example, a hypothetical set of values might be:

```text
Cell:         0    1    2    3    4    5    6    7    8
Q-value:     0.1  0.3  0.2  0.0  0.8  0.4  0.2  0.1  0.3
```

The agent would prefer cell 4, provided it is empty, because its estimated value is the highest.

These numbers are only an illustration; the actual values are learned during training.

**Why nine values?**

A Tic-Tac-Toe board contains nine cells, so there are at most nine possible actions before considering which cells are already occupied.

### What is `alpha`?

```python
alpha = 0.1
```

`alpha` is the **learning rate**.

It controls how much the agent changes its old Q-value when it receives new information.

- A larger alpha means the agent updates its values more strongly.
- A smaller alpha means the agent updates its values more gradually.

Here, `0.1` means the agent makes a relatively small update each time.

### What is `gamma`?

```python
gamma = 0.9
```

`gamma` is the **discount factor**.

It controls how much the agent values future rewards.

For example, a move might not win immediately, but it might create a strong position that leads to a win later.

A high gamma means the agent gives considerable importance to future rewards.

Our gamma is `0.9`, so future rewards matter quite a lot.

### What is `epsilon`?

```python
epsilon = 1.0
```

Epsilon controls **exploration**.

An agent has two broad choices:

- **Explore:** Try a move to gain experience.
- **Exploit:** Choose the move that currently appears best according to what it has learned.

At the beginning, the agent knows very little. So we start with `epsilon = 1.0`, which makes it explore by choosing random moves.

During training, epsilon gradually decreases. This encourages the agent to use its learned knowledge more often.

**Remember these three parameters:**

| Parameter | Meaning | Purpose |
|---|---|---|
| `alpha` | Learning rate | Controls the size of learning updates |
| `gamma` | Discount factor | Controls the importance of future rewards |
| `epsilon` | Exploration rate | Controls how often the agent explores |

---

# Part B: Defining the Tic-Tac-Toe game

## 1. Representing the board

```python
board = [" "] * 9
```

This creates a list containing nine empty cells.

Initially, it looks like this:

```text
[" ", " ", " ", " ", " ", " ", " ", " ", " "]
```

We represent the board using these positions:

```text
  0 | 1 | 2
 ---+---+---
  3 | 4 | 5
 ---+---+---
  6 | 7 | 8
```

For example:

```python
board[4] = "X"
```

places X in the middle.

The board then looks like this:

```text
   |   |  
---+---+---
   | X |  
---+---+---
   |   |  
```

Python uses zero-based indexing, meaning the first position is index `0`, not index `1`.

## 2. Checking whether a player has won

```python
def check_winner(board, mark):
    wins = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]

    return any(all(board[i] == mark for i in line) for line in wins)
```

Let's understand the important concepts.

### What is a function?

A function is a reusable block of code that performs a task.

We define a function using `def`.

For example:

```python
def greet():
    print("Hello")
```

We can run it using:

```python
greet()
```

### What does `check_winner(board, mark)` do?

It checks whether the specified player has won.

- `board` contains the current board.
- `mark` is the player's symbol: `"X"` or `"O"`.

For example:

```python
check_winner(board, "X")
```

checks whether X has won.

### What is the `wins` list?

There are eight possible winning lines in Tic-Tac-Toe:

- Three horizontal lines.
- Three vertical lines.
- Two diagonal lines.

For example:

```python
(0, 1, 2)
```

represents the top row.

If all three positions contain X, X has won.

### What does `all()` do?

`all()` returns `True` when every condition in a collection is true.

For example:

```python
all(board[i] == "X" for i in (0, 1, 2))
```

checks whether all three cells in the top row contain X.

### What does `any()` do?

`any()` returns `True` if at least one condition is true.

Our function checks all eight winning lines. If any line contains the same player's mark in all three cells, the player wins.

You don't need to memorize the entire expression. Remember:

**`check_winner()` checks whether X or O has three marks in a winning line.**

## 3. Finding available moves

```python
def available_moves(board):
    return [i for i in range(9) if board[i] == " "]
```

This function finds the empty cells.

Suppose the board is:

```text
 X |   | O
---+---+---
   | X |  
---+---+---
 O |   |  
```

The occupied cells are 0, 2, 4, and 6.

The available moves are:

```text
[1, 3, 5, 7, 8]
```

### Understanding the expression

- `range(9)` generates the indexes from 0 to 8.
- `board[i] == " "` checks whether a cell is empty.
- The list comprehension collects only the indexes of empty cells.

This function is important because the agent must not place a mark in an occupied cell.

## 4. Displaying the board

```python
def show_board(board):
    for i in range(0, 9, 3):
        print(" | ".join(board[i:i+3]))
        if i < 6:
            print("--+---+--")
```

This function prints the board in a readable format.

However, **the current training and testing code does not call `show_board()`**. It is included as a helper if you later want to display a game.

You don't need to understand every detail of string formatting for the reinforcement learning concepts.

---

# Part C: Building the reinforcement learning model

This is the most important section of the practical.

## 1. Choosing an action

```python
def choose_action(state, moves, explore=True):
    if explore and random.random() < epsilon:
        return random.choice(moves)

    values = q_table[state]
    best_value = max(values[i] for i in moves)
    best_moves = [i for i in moves if values[i] == best_value]
    return random.choice(best_moves)
```

The purpose of this function is to choose the next move.

### What is `state`?

The state is the current board configuration.

We represent it as a tuple, for example:

```python
state = tuple(board)
```

A tuple is similar to a list, but it cannot be modified after creation. It allows the board configuration to be used as a dictionary key in the Q-table.

### What is `moves`?

`moves` contains the empty cell positions.

The agent must choose from these positions only.

### What does `explore=True` mean?

It allows the function to use epsilon to decide whether to explore.

During training, we call:

```python
choose_action(state, moves)
```

The default `explore=True` is used, so the agent can explore.

During testing, we call:

```python
choose_action(state, moves, explore=False)
```

This prevents the agent from deliberately choosing random moves for exploration. It instead selects a move with the highest learned Q-value among the available cells.

### How does exploration work?

```python
if explore and random.random() < epsilon:
    return random.choice(moves)
```

`random.random()` generates a random number between 0 and 1.

If that number is less than epsilon, the agent chooses a random available move.

For example, if epsilon is `0.8`, there is an 80% chance of exploring.

Otherwise, it uses the learned Q-values to choose a move.

### How does the agent exploit its knowledge?

```python
values = q_table[state]
best_value = max(values[i] for i in moves)
best_moves = [i for i in moves if values[i] == best_value]
return random.choice(best_moves)
```

These lines:

1. Retrieve the Q-values for the current board.
2. Find the highest Q-value among the available moves.
3. Collect the moves tied for the highest value.
4. Randomly select one of those tied moves.

**Why consider only available moves?**

Because choosing the highest-value cell without checking whether it is empty could result in an illegal move.

## 2. Updating the Q-table

```python
def update_q(state, action, reward, next_state=None, next_moves=None):
    old_value = q_table[state][action]

    if next_state is None:
        future_value = 0
    else:
        future_value = max(q_table[next_state][i] for i in next_moves)

    q_table[state][action] = old_value + alpha * (
        reward + gamma * future_value - old_value
    )
```

This function is where the agent actually learns.

Let's break it down.

### What information does it receive?

- `state`: the board before the agent's move.
- `action`: the move the agent selected.
- `reward`: the immediate reward for the outcome.
- `next_state`: the board after the opponent responds, if the game is continuing.
- `next_moves`: the available moves in that next state.

### What is `old_value`?

```python
old_value = q_table[state][action]
```

This retrieves the agent's current estimate of how useful its chosen action is in the current state.

### What is `future_value`?

```python
if next_state is None:
    future_value = 0
else:
    future_value = max(q_table[next_state][i] for i in next_moves)
```

If the game has ended, there are no future rewards to estimate, so the future value is set to zero.

Otherwise, the agent looks at the next state and finds the highest estimated value among the available moves.

This lets the agent learn that an action can be useful because it leads to another state with promising future moves.

### Understanding the Q-learning formula

The code uses this formula:

**New Q-value = Old Q-value + alpha × (Reward + gamma × Future Value − Old Q-value)**

You don't need to derive the formula for this practical. Understand what it means:

- `Old Q-value`: what the agent previously believed.
- `Reward`: the immediate feedback it received.
- `Future Value`: the estimated value of the next state.
- `alpha`: how strongly the agent updates its belief.
- `gamma`: how much it values the future.

The expression inside the brackets measures the difference between the new target estimate and the old estimate.

The agent uses that difference to improve its Q-value.

**Remember:** Q-learning updates the estimated usefulness of actions based on rewards and estimates of future rewards.

---

# Part D: Training the model

```python
for episode in range(20000):
    board = [" "] * 9
```

### What is an episode?

An episode is one complete game, from an empty board until someone wins or the game ends in a draw.

Here:

```python
range(20000)
```

means the agent will play up to 20,000 training games.

Why so many?

One game is usually not enough to learn useful behaviour. Repeated games allow the agent to experience different board states and outcomes.

### Step 1: Record the current state

```python
state = tuple(board)
moves = available_moves(board)
```

The agent records the current board and finds all legal moves.

### Step 2: The agent chooses and makes a move

```python
action = choose_action(state, moves)
board[action] = "X"
```

First, the agent chooses a cell.

Then the code places X in that cell.

### Step 3: Check whether X wins

```python
if check_winner(board, "X"):
    update_q(state, action, 1)
    break
```

If X wins, the agent receives a reward of `+1`.

The Q-table is updated, and `break` ends the current game.

This gives the agent positive feedback for a winning move.

### Step 4: Check whether the game is a draw

```python
if not available_moves(board):
    update_q(state, action, 0)
    break
```

If there are no empty cells and nobody has won, the game is a draw.

The agent receives a reward of `0`.

### Step 5: The opponent makes a move

```python
board[random.choice(available_moves(board))] = "O"
```

The opponent chooses a random empty cell and places O there.

This gives the agent experience against a simple opponent.

### Step 6: Check whether O wins

```python
if check_winner(board, "O"):
    update_q(state, action, -1)
    break
```

If O wins, the agent receives a reward of `-1`.

This penalizes the action that preceded the opponent's winning response.

### Step 7: Update the Q-value if the game continues

```python
next_state = tuple(board)
next_moves = available_moves(board)
update_q(state, action, 0, next_state, next_moves)
```

If nobody has won and the game is not a draw:

1. The program records the board after O's response.
2. It identifies the available moves in that state.
3. It updates the Q-value for the preceding X action.

The immediate reward is `0`, but the update also considers the estimated value of the next state.

This is how the agent can learn from sequences of moves, rather than only from immediate wins.

### Step 8: Reduce exploration

```python
epsilon = max(0.05, epsilon * 0.9998)
```

After each training game, epsilon decreases slightly.

Initially, the agent explores frequently. As training continues, it increasingly uses what it has learned.

The `max()` function ensures epsilon never falls below `0.05`.

This means exploration is reduced, but it never falls completely to zero during training.

**One limitation to understand:** The opponent is random, not an optimal Tic-Tac-Toe player. Training against a random opponent does not guarantee that the agent will play perfectly or defeat a strong human player.

---

# Part E: Testing the model

After training, we want to check how the agent performs on games it has not played during training.

```python
wins = 0
losses = 0
draws = 0
```

These variables count the outcomes.

Next:

```python
for game in range(1000):
    board = [" "] * 9
```

This starts 1,000 test games, each with an empty board.

The testing code follows the same game rules as training, but the agent chooses its moves using:

```python
action = choose_action(state, moves, explore=False)
```

Setting `explore=False` prevents the agent from deliberately making random exploratory choices.

It selects the best currently estimated action instead.

After each game, the program records whether the agent won, lost, or drew.

Finally:

```python
print("Results after 1000 test games:")
print("Wins:", wins)
print("Losses:", losses)
print("Draws:", draws)
print("Win percentage:", round(wins / 1000 * 100, 2), "%")
```

The win percentage is calculated as:

**Win percentage = (Number of wins / Total test games) × 100**

For example, if the agent wins 600 of the 1,000 test games:

Win percentage = (600 / 1000) × 100 = 60%

That is just an example, not a prediction of your actual output.

### How do we know whether training worked?

Look at the results:

- A high win percentage suggests the agent performs well against the random opponent.
- Losses show that the agent still makes moves that allow the opponent to win.
- Draws show that neither player won those games.

For a fairer evaluation, compare the trained agent's results with a completely random agent under the same test conditions.

---

## 5. VIVA CONCEPT CRASH COURSE

Here are the concepts to remember for your exam.

| Term | Simple meaning |
|---|---|
| Reinforcement Learning | Learning by taking actions and receiving rewards or penalties |
| Agent | The learner or decision-maker |
| Environment | The world in which the agent acts |
| State | The current situation or board |
| Action | A move selected by the agent |
| Reward | Feedback for an outcome |
| Q-table | Stores estimated values of actions in different states |
| Q-learning | A method that learns action values from rewards and future estimates |
| Episode | One complete game |
| Exploration | Trying actions to gain experience |
| Exploitation | Choosing an action based on learned values |
| Learning rate (`alpha`) | Controls the size of Q-value updates |
| Discount factor (`gamma`) | Controls the importance of future rewards |
| Epsilon (`epsilon`) | Controls the probability of exploration |

### TOP 3 EXPECTED VIVA QUESTIONS

**1. What is reinforcement learning?**

Reinforcement learning is a type of machine learning in which an agent learns by interacting with an environment and receiving rewards or penalties.

**2. What is Q-learning?**

Q-learning is a reinforcement learning algorithm that learns the estimated value of taking different actions in different states.

**3. What are the rewards used in this Tic-Tac-Toe program?**

- Win: `+1`
- Loss: `-1`
- Draw: `0`

---

## Final revision: Understand the whole practical in six lines

1. **Environment:** We create a Tic-Tac-Toe board with nine cells.
2. **Agent:** The computer plays as X, while O makes random moves.
3. **Model:** The Q-table stores the estimated values of moves for different board states.
4. **Training:** The agent plays 20,000 games and updates its Q-values based on rewards and future estimates.
5. **Testing:** The trained agent plays 1,000 games without deliberate exploration.
6. **Evaluation:** We count wins, losses, and draws to measure its performance.

**The single most important idea:** The agent is not given a list of correct moves. It learns which moves are useful through repeated games and feedback.