Formative 3 Assignment: Deep Q Learning
Due Sunday by 11:59pm Points 30 Submitting a website url or a file upload File Types zip Attempts 0 Allowed Attempts 2 Available after Mar 9 at 12am
In this assignment, you will use Stable Baselines 3 and Gymnasium to train and evaluate a Deep Q-network (DQN) agent. Your goal is to train the agent to play an Atari game and then evaluate its performance by running it in the same environment.

Environment Selection
Use an Atari environment from the Gymnasium collection. You can choose any environment from the list here:
Atari EnvironmentsLinks to an external site.

You will implement two Python scripts:

train.py: To train the agent and save its policy network.
play.py: To load the trained model and play with the agent.
Task 1: Training Script (train.py)
Objective:
Create a Python script, train.py, that will train a DQN agent to play your selected Atari environment using Stable Baselines3.

Instructions:

Define the Agent:

Use Stable Baselines3 to define a DQN agent.
Compare MLPPolicy (Multilayer Perceptron) and CNNPolicy (Convolutional Neural Network) policies to see which performs better for your chosen environment.
Train the Agent:

Train your DQN agent in the chosen Atari environment.
Save the trained model as dqn_model.zip.
Log key training details such as:
Reward trends
Episode length
Hyperparameter Tuning:

Tune the following hyperparameters and record the observed behavior in a table:
Learning Rate (lr)
Gamma (γ): Discount factor.
Batch Size (batch_size): Number of experiences sampled from memory for each update step.
Epsilon (epsilon_start, epsilon_end, epsilon_decay): Controls exploration in ε-greedy policies.
Hyperparameter Set Documentation:

Record the hyperparameter configurations you tested in the following table (add more rows for additional experiments):
NB: Each GROUP MEMBER MUST EXPERIMENT WITH 10 Experiments ( 10 Different values for the hyperparameters, meaning 10 different combinations of hyperparameter values)
MEMBER NAME

 

Hyperparameter Set 

Noted Behavior

lr=, gamma=, batch= , epsilon_start, epsilon_end, epsilon_decay

lr=, gamma=, batch=, epsilon_start, epsilon_end, epsilon_decay

Add 8 More Rows..........

 

Task 2: Playing Script (play.py)
Objective:
Create a Python script, play.py, that loads the trained model and plays with the agent in the Atari environment.

Instructions:

Load the best Trained Model:

Use DQN.load('/path/to/dqn_model.zip') to load the saved model.
Set Up the Environment:

Use the same environment as you did in train.py (i.e., Atari).
Use GreedyQPolicy:

For evaluation, use GreedyQPolicy, which ensures the agent selects actions with the highest Q-value (maximizing performance).
Display the Game:

Run a few episodes and visualize the agent’s performance in real time.
Use env.render() to display the game on a GUI.
Task 3: Group Presentations - 10Minutes
Cameras must remain ON for the entire presentation.

Presentation Flow

Introduce the Atari Env Used
Each member will:

Briefly introduce the 10 hyperparameter tuning experiments they conducted.
Highlight key insights from their tuning:
Which hyperparameter changes improved performance?
Which changes harmed performance?
What final configuration performed best and why?
Time Limit: Maximum 2 minutes per member (strict)

The presentation should focus on decision-making and results rather than code walkthroughs.

Finally, the group will show a short gameplay clip (play.py output).

Q&A
After the group finishes their individual insights, the Coach will conduct a short Q&A session with the team.

All members must be prepared to answer questions about:

Trade-offs in hyperparameter choices
Why does their final model behave the way it does
Policy architecture decision (MLP vs CNN)
Expected Artifacts During Presentation
Display your hyperparameter table.
Show a short gameplay clip (play.py output).
Submission Instructions:
Attempt 1: Submit a zip file of your GitHub Repository containing both the train.py and play.py scripts.
Attempt 2: Submit a GitHub Repository URL with both the train.py and play.py scripts and all necessary files (including zip policy).
Readme must have the table and a video showing the agent playing in the Atari environment (showing the play.py script running and the agent interacting with the environment).
Discuss hyperparameter tuning results based on the table provided
Note: Every Group Must book a Slot with the Coach in Week 6

Link to SlotsLinks to an external site.

Rubric
Formative 2 - DQN (ATARI)
Formative 2 - DQN (ATARI)
Criteria	Ratings	Pts
This criterion is linked to a Learning OutcomeUnderstanding of Deep Q Learning (DQN) and RL Concepts
Evaluated in the Presentation , Questions posed by the Coach and students answer
10 to >6.0 pts
Excemplary
The student demonstrates a thorough understanding of DQN, RL concepts, exploration-exploitation tradeoffs, and reward structures.
6 to >4.0 pts
Proficient
The student shows a good understanding with only minor gaps or unclear details.
4 to >0.0 pts
Developing
The student shows limited understanding with major gaps or misconceptions.
0 pts
Beginning
The student demonstrates little to no understanding of DQN and RL concepts.
10 pts
This criterion is linked to a Learning OutcomeHyperparameter Tuning and Documentation
5 to >3.0 pts
Excemplary
Hyperparameter configurations are tested and thoroughly documented. Observations clearly explain how each configuration impacts performance.
3 to >2.0 pts
Proficient
Hyperparameter configurations are tested and documented, but explanations are less detailed or incomplete.
2 to >0.0 pts
Developing
Hyperparameter configurations are either missing or poorly documented. No clear explanation of their effects.
0 pts
Beginning
Hyperparameter tuning is missing or entirely incorrect.
5 pts
This criterion is linked to a Learning OutcomeEvaluation and Agent Performance in play.py
5 to >3.0 pts
Excemplary
The play.py script works as expected, loading the trained model and displaying the game. The agent’s performance is strong and the game is rendered correctly.
3 to >2.0 pts
Proficient
The play.py script mostly works, but there are minor issues with agent performance or game display.
2 to >0.0 pts
Developing
The play.py script works with issues, and the agent’s performance is suboptimal or the game display does not work.
0 pts
Beginning
The play.py script does not function, or the agent’s performance is very poor.
5 pts
This criterion is linked to a Learning OutcomeGroup Collaboration and Individual Contribution
10 to >6.0 pts
Excemplary
Group Collaboration and Individual Contribution Individual contributions are documented, with a well-distributed workload and evident collaboration among members.
6 to >4.0 pts
Proficient
Group contributions are mostly documented, but the division of work or collaboration could be clearer.
4 to >0.0 pts
Developing
Contributions are unclear or the work distribution is uneven among group members.
0 pts
Beginning
There is little to no collaboration or clear division of tasks among group members.
10 pts
Total Points: 30
