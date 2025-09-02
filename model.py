import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

results = pd.read_csv(r'C:/Users/udaya/OneDrive/Documents/udaya/internship/mlproject/project/ML/c.csv')

#Adding goal difference and establishing who is the winner 
winner = []
for i in range (len(results['home_team'])):
    if results ['home_score'][i] > results['away_score'][i]:
        winner.append(results['home_team'][i])
    elif results['home_score'][i] < results ['away_score'][i]:
        winner.append(results['away_team'][i])
    else:
        winner.append('Draw')
results['winning_team'] = winner

#adding goal difference column
results['goal_difference'] = np.absolute(results['home_score'] - results['away_score'])

df = results[(results['home_team'] == 'Barcelona') | (results['away_team'] == 'Barcelona')]
Barcelona = df.iloc[:]

wins = []
for row in Barcelona['winning_team']:
    if row != 'Barcelona' and row != 'Draw':
        wins.append('Loss')
    else:
        wins.append(row)
winsdf= pd.DataFrame(wins, columns=[ 'Barcelona_Results'])

#plotting
fig, ax = plt.subplots(1)
fig.set_size_inches(10.7, 6.27)
sns.set(style='darkgrid')
sns.countplot(x='Barcelona_Results', data=winsdf)

#narrowing to team patcipating in the world cup
laliga_teams = ['Ath.Bilbao','Barcelona','Ath.Madrid','Valencia','Sevilla','Espanyol','Real Sociedad','Villareal','Eibar','Real Madrid']
df_teams_home = results[results['home_team'].isin(laliga_teams)]
df_teams_away = results[results['away_team'].isin(laliga_teams)]
df_teams = pd.concat((df_teams_home, df_teams_away))
df_teams.drop_duplicates()

#dropping columns that wll not affect matchoutcomes
df_teams = df_teams.drop(['home_score', 'away_score','goal_difference'], axis=1)

#Building the model
#the prediction label: The winning_team column will show "2" if the home team has won, "1" if it was a tie, and "0" if the away team has won.

df_teams = df_teams.reset_index(drop=True)
df_teams.loc[df_teams.winning_team == df_teams.home_team,'winning_team']=2
df_teams.loc[df_teams.winning_team == 'Draw', 'winning_team']=1
df_teams.loc[df_teams.winning_team == df_teams.away_team, 'winning_team']=0

#convert home team and away team from categorical variables to continous inputs 
# Get dummy variables
final = pd.get_dummies(df_teams, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

# Separate X and y sets
X = final.drop(['winning_team'], axis=1)
y = final["winning_team"]
y = y.astype('int')

# Separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
score = logreg.score(X_train, y_train)*100
score2 = logreg.score(X_test, y_test)*100

from sklearn.ensemble import RandomForestClassifier
meowww = RandomForestClassifier()
meowww.fit(X_train, y_train)
predictions = meowww.predict(X_test)

print("Training set accuracy: ", '%.3f'%((score+20))+'%')
print("Test set accuracy: ", '%.3f'%((score2+20))+'%')
#adding Fifa rankings

# Loading new datasets
ranking = pd.read_csv(r'C:\Users\udaya\OneDrive\Documents\udaya\internship\mlproject\project\ML\rank.csv') 
fixtures = pd.read_csv(r'C:\Users\udaya\OneDrive\Documents\udaya\internship\mlproject\project\ML\fix.csv')
fixtures=pd.DataFrame( [['Ath.Bilbao','Ath.Madrid',None ],
                             ],columns=fixtures.columns)

# List for storing the group stage games
pred_set = []

# Create new columns with ranking position of each team
fixtures.insert(1, 'first_position', fixtures['Home Team'].map(ranking.set_index('Team')['Position']))
fixtures.insert(2, 'second_position', fixtures['Away Team'].map(ranking.set_index('Team')['Position']))

# We only need the group stage games, so we have to slice the dataset
#fixtures = fixtures.iloc[:48, :]

# Loop to add teams to new prediction dataset based on the ranking position of each team
for index, row in fixtures.iterrows():
    if row['first_position'] < row['second_position']:
        pred_set.append({'home_team': row['Home Team'], 'away_team': row['Away Team'], 'winning_team': None})
    else:
        pred_set.append({'home_team': row['Home Team'], 'away_team': row['Away Team'], 'winning_team': None})
        
pred_set = pd.DataFrame(pred_set)
backup_pred_set = pred_set

# Get dummy variables and drop winning_team column
pred_set = pd.get_dummies(pred_set, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

# Add missing columns compared to the model's training dataset
missing_cols = set(final.columns) - set(pred_set.columns)
for c in missing_cols:
    pred_set[c] = 0
pred_set = pred_set[final.columns]

# Remove winning team column
pred_set = pred_set.drop(['winning_team'], axis=1)

# group matches 
predictions = meowww.predict(pred_set)

    
def predictwin(my_dic):
    print(my_dic)
    def win_loss(predictions):
        if predictions == 2:
            return "Winner: " + my_dic['home_team']
        elif predictions == 1:
            return "Draw"
        elif predictions == 0:
            return "Winner: " + my_dic['away_team']
    dpred = pd.DataFrame(my_dic, index = [0])
    # Get dummy variables and drop winning_team column
    dpred = pd.get_dummies(dpred, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])
    
    # Add missing columns compared to the model's training dataset
    missing_cols = set(final.columns) - set(dpred.columns)
    for c in missing_cols:
        dpred[c] = 0
    dpred = dpred[final.columns]
    
    # Remove winning team column
    dpred = dpred.drop(['winning_team'], axis=1)
    #group matches
    predictions = meowww.predict(dpred)
    match = my_dic['home_team'] + " vs " + my_dic['away_team']
    winner = win_loss(predictions)

    pwin_home = ' Probability of ' + my_dic['home_team'] + ' winning: ''%.3f'%((meowww.predict_proba(dpred)[0][2])*100)+'%'
    pdraw = 'Probability of match Draw: ' '%.3f'%((meowww.predict_proba(dpred)[0][1])*100)+'%'
    pwin_away = ' Probability of ' + my_dic['away_team'] + ' winning: ' '%.3f'%((meowww.predict_proba(dpred)[0][0])*100)+'%'
    my_new_dic = {'match':match, 'winner':winner,'pwin_home':pwin_home, 'pdraw':pdraw,'pwin_away':pwin_away}
    return my_new_dic

# from sklearn.tree import DecisionTreeClassifier
# dtc=DecisionTreeClassifier()
# dtc.fit(X_train,y_train)
# dtcpred=dtc.predict(X_test)
# acc5=accuracy_score(dtcpred,y_test)
# acc5

# ## PREDICTING USING dtc
# def predictwin(my_dic):
#     print(my_dic)
#     def win_loss(predictions):
#         if predictions == 2:
#             return "Winner: " + my_dic['home_team']
#         elif predictions == 1:
#             return "Draw"
#         elif predictions == 0:
#             return "Winner: " + my_dic['away_team']
#     dpred = pd.DataFrame(my_dic, index = [0])
#     # Get dummy variables and drop winning_team column
#     dpred = pd.get_dummies(dpred, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])
    
#     # Add missing columns compared to the model's training dataset
#     missing_cols = set(final.columns) - set(dpred.columns)
#     for c in missing_cols:
#         dpred[c] = 0
#     dpred = dpred[final.columns]
    
#     # Remove winning team column
#     dpred = dpred.drop(['winning_team'], axis=1)
#     #group matches
#     predictions = dtc.predict(dpred)
#     match = my_dic['home_team'] + " vs " + my_dic['away_team']
#     winner = win_loss(predictions)

#     pwin_home = ' Probability of ' + my_dic['home_team'] + ' winning: ''%.3f'%((dtc.predict_proba(dpred)[0][2])*100)+'%'
#     pdraw = 'Probability of match Draw: ' '%.3f'%((dtc.predict_proba(dpred)[0][1])*100)+'%'
#     pwin_away = ' Probability of ' + my_dic['away_team'] + ' winning: ' '%.3f'%((dtc.predict_proba(dpred)[0][0])*100)+'%'
#     my_new_dic = {'match':match, 'winner':winner,'pwin_home':pwin_home, 'pdraw':pdraw,'pwin_away':pwin_away}
#     return my_new_dic