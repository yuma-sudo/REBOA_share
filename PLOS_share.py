# Loading all needed packages
import pandas as pd
import sys
from julia import Julia
Julia(sysimage='/home/gridsan/groups/IAI/images/2.2.0/julia-1.6.1/sys.so', compiled_modules = False)
from interpretableai import iai
from pandas.api.types import CategoricalDtype
from datetime import datetime
from pytz import timezone
tz = timezone('EST')
print(datetime.now(tz))
import numpy as np


# Load data
df = pd.read_csv('REBOA_data.csv')
print('loaded data')

# Conduct Data Cleaning and Preprocessing
df.sex = df.sex.astype('category')
df.supplementaloxygen = df.supplementaloxygen.astype('category')
df.teachingstatus = df.teachingstatus.astype('category')
df.ed_sol = df.ed_sol.astype('category')
df.intubated = df.intubated.astype('category')
df.chest_tube_1hr = df.chest_tube_1hr.astype('category')

df.transf_rbc_1hr = df.transf_rbc_1hr.astype('category')
df.transf_wholeblood_1hr = df.transf_wholeblood_1hr.astype('category')
df.pelvic_fx = df.pelvic_fx.astype('category')
df.femur_fx = df.femur_fx.astype('category')

df.hemothorax = df.hemothorax.astype('category')
df.pneumothorax = df.pneumothorax.astype('category')
df.thoracic_aorta_inj = df.thoracic_aorta_inj.astype('category')
df.hemoperitoneum = df.hemoperitoneum.astype('category')

df['pulserate'] = (df['pulserate'] < 60).astype(int)
df['bmi'] = (df['bmi'] < 25).astype(int)

# Define Co-variables and Outcomes
X_col = ['age', 'sex', 'sbp', 'pulserate', 'temperature', 'gcs',
       'respiratoryrate', 'pulseoximetry', 'supplementaloxygen', 'intubated',
       'height', 'weight', 'bmi', 'ed_sol', 'teachingstatus',
       'verificationlevel', 'chest_tube_1hr', 'transf_rbc_1hr',
       'transf_wholeblood_1hr', 'pelvic_fx', 'femur_fx', 'hemothorax',
       'pneumothorax', 'thoracic_aorta_inj', 'hemoperitoneum']
X = df[X_col]
T = df[['i_reboa_4hr', 'i_ed_thoracotomy_1hr']]
Y = df.o_mortality_24hr

# Define train and test sets
(train_X, train_treatments, train_outcomes), (test_X, test_treatments, test_outcomes) = (
    iai.split_data('policy_minimize', X, T, Y, seed=123, train_proportion=0.5))

# Conduct imputation on missing data
lnr = iai.ImputationLearner(method='opt_knn', random_seed=1)
train_X = lnr.fit_transform(train_X)
test_X = lnr.transform(test_X)

print('finished imputation')

# For specific values, truncate values to avoid numerical precision instabilities
for i in ['age', 'sbp', 'pulserate', 'gcs', 'respiratoryrate', 'pulseoximetry', 
          'verificationlevel']:
    train_X[i] = round(train_X[i])
    test_X[i] = round(test_X[i])

for i in ['height', 'weight']:
    train_X[i] = round(train_X[i], 3)
    test_X[i] = round(test_X[i], 3)

for i in ['temperature']:
    train_X[i] = round(train_X[i], 2)
    test_X[i] = round(test_X[i], 2)

train_treatments['combined'] = list(map(str, train_treatments.values))
train_treatments = train_treatments.combined

test_treatments['combined'] = list(map(str, test_treatments.values))
test_treatments = test_treatments.combined

# Compute rewards for train and test
reward_lnr = iai.CategoricalClassificationRewardEstimator(
    propensity_estimator=iai.RandomForestClassifier(),
    outcome_estimator=iai.RandomForestClassifier(),
    reward_estimator='direct_method',
    random_seed=123,
)
train_rewards, train_reward_score = reward_lnr.fit_predict(
    train_X, train_treatments, train_outcomes,
    propensity_score_criterion='auc', outcome_score_criterion='auc')

print('finished training reward estimation')
print(train_reward_score)

# Fit OPT models
grid = iai.GridSearch(
iai.OptimalTreePolicyMinimizer(
    random_seed=121),
    minbucket=[10, 50, 100],
    max_depth=range(3, 7),
)
grid.fit(train_X, train_rewards, train_proportion=0.5)

test_reward_lnr = iai.CategoricalClassificationRewardEstimator(
    propensity_estimator=iai.RandomForestClassifier(),
    outcome_estimator=iai.RandomForestClassifier(),
    reward_estimator='direct_method',
    random_seed=1,
)

test_rewards, test_reward_score = test_reward_lnr.fit_predict(
    test_X, test_treatments, test_outcomes,
    propensity_score_criterion='auc', outcome_score_criterion='auc')

print('finished testing reward estimation')
print(test_reward_score)

# Evaluate outcomes
policy_outcomes = grid.predict_outcomes(test_X, test_rewards)
now = policy_outcomes.mean()

original = np.mean([test_rewards[str(test_treatments[i])][i] for i in range(len(test_treatments))])
print('original', original)
print('now', now)
print('ARR', original - now)
print('RRR', (original - now)/original)

# Write and save the final models
grid.get_learner().write_html(f'tree.html')
grid.get_learner().write_html(f'tree.json')

# Further evaluation of the models
test_rewards_reboa = test_rewards[test_treatments  == 1]
test_rewards_no_reboa = test_rewards[test_treatments  == 0]

test_X_reboa = test_X[test_treatments  == 1]
test_X_no_reboa = test_X[test_treatments  == 0]

test_treatments_reboa = test_treatments[test_treatments == 1]
test_treatments_no_reboa = test_treatments[test_treatments == 0]

policy_outcomes = grid.predict_outcomes(test_X_reboa, test_rewards_reboa)
now = policy_outcomes.mean()

original = np.mean([test_rewards_reboa[str(test_treatments_reboa[i])].values[i] for i in range(len(test_treatments_reboa))])

print('ARR_REBOA', (original - now)*100)
print('RRR_REBOA', (original - now)/original*100)

policy_outcomes = grid.predict_outcomes(test_X_no_reboa, test_rewards_no_reboa)
now = policy_outcomes.mean()

original = np.mean([test_rewards_no_reboa[str(test_treatments_no_reboa[i])].values[i] for i in range(len(test_treatments_no_reboa))])

print('ARR_NO_REBOA', (original - now)*100)
print('RRR_NO_REBOA', (original - now)/original*100)

print(datetime.now(tz))
