import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('diabetes.csv')
# print(data.info())

columnsDictionary = {0: 'Pregnancies', 1: 'Glucose', 2: 'BloodPressure', 3: 'SkinThickness', 4: 'Insulin', 5: 'BMI',
                     6: 'DiabetesPedigreeFunction', 7: 'Age'}

pregnancies = ctrl.Antecedent(np.arange(data['Pregnancies'].min(), data['Pregnancies'].max(), 1), 'Pregnancies')
glucose = ctrl.Antecedent(np.arange(data['Glucose'].min(), data['Glucose'].max(), 1), 'Glucose')
bloodPressure = ctrl.Antecedent(np.arange(data['BloodPressure'].min(), data['BloodPressure'].max(), 1), 'BloodPressure')
skinThickness = ctrl.Antecedent(np.arange(data['SkinThickness'].min(), data['SkinThickness'].max(), 1), 'SkinThickness')
insulin = ctrl.Antecedent(np.arange(data['Insulin'].min(), data['Insulin'].max(), 1), 'Insulin')
bmi = ctrl.Antecedent(np.arange(data['BMI'].min(), data['BMI'].max(), 0.01), 'BMI')
diabetesPedigreeFunction = ctrl.Antecedent(np.arange(data['DiabetesPedigreeFunction'].min(), data['DiabetesPedigreeFunction'].max(), 0.01), 'DiabetesPedigreeFunction')
age = ctrl.Antecedent(np.arange(data['Age'].min(), data['Age'].max(), 1), 'Age')
# outcome = ctrl.Antecedent(np.arange(data['Outcome'].min(), data['Outcome'].max(), 1), 'Outcome')

# pregnancies = ctrl.Antecedent(np.arange(0, 18, 1), 'Pregnancies')
# glucose = ctrl.Antecedent(np.arange(0, 201, 1), 'Glucose')
# bloodPressure = ctrl.Antecedent(np.arange(0, 121, 1), 'BloodPressure')
# skinThickness = ctrl.Antecedent(np.arange(0, 81, 1), 'SkinThickness')
# insulin = ctrl.Antecedent(np.arange(0, 321, 1), 'Insulin')
# bmi = ctrl.Antecedent(np.arange(0, 68, 0.01), 'BMI')
# diabetesPedigreeFunction = ctrl.Antecedent(np.arange(0, 2.51, 0.01), 'DiabetesPedigreeFunction')
# age = ctrl.Antecedent(np.arange(21, 82, 1), 'Age')

columns = [pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]
for column in columns:
    column.automf(3)
    # column.view()

obj = ctrl.Consequent(np.arange(0, 2, 1), 'Outcome')
obj['no_diabetes'] = fuzz.trimf(obj.universe, [0, 0, 1])
obj['diabetes'] = fuzz.trimf(obj.universe, [0, 1, 1])
# obj.view()

x_train, x_test, y_train, y_test = train_test_split(data.drop('Outcome', axis=1), data['Outcome'], test_size=0.2)

pred = []
# poor, mediocre, average, decent, good

# rule1 = ctrl.Rule(pregnancies['good'] & glucose['good'], obj['poor'])
# rule2 = ctrl.Rule(bloodPressure['good'] & skinThickness['good'], obj['poor'])
# rule3 = ctrl.Rule(insulin['average'] & bmi['average'], obj['good'])
# rule4 = ctrl.Rule(diabetesPedigreeFunction['average'] & age['average'], obj['average'])
# rule5 = ctrl.Rule(glucose['good'] & bloodPressure['good'], obj['good'])
# rule6 = ctrl.Rule(skinThickness['poor'] & insulin['good'], obj['poor'])
# rule7 = ctrl.Rule(bmi['good'] & diabetesPedigreeFunction['good'] & bmi['good'], obj['good'])
# rule8 = ctrl.Rule(age['poor'] & pregnancies['good'], obj['poor'])

rule1 = ctrl.Rule(glucose['good'] & pregnancies['good'] | insulin['good'] | bmi['good'], obj['diabetes'])
rule2 = ctrl.Rule(bloodPressure['good'] & skinThickness['good'], obj['diabetes'])
rule3 = ctrl.Rule(insulin['average'] & bmi['average'], obj['no_diabetes'])
rule4 = ctrl.Rule(diabetesPedigreeFunction['average'] & age['average'], obj['no_diabetes'])
rule5 = ctrl.Rule(glucose['good'] & bloodPressure['good'], obj['no_diabetes'])
rule6 = ctrl.Rule(skinThickness['poor'] & insulin['good'], obj['diabetes'])
rule7 = ctrl.Rule(bmi['good'] & diabetesPedigreeFunction['good'] & insulin['good'], obj['no_diabetes'])
rule8 = ctrl.Rule(age['poor'] & pregnancies['good'], obj['diabetes'])
rule9 = ctrl.Rule(pregnancies['good'] & glucose['good'] & age['average'], obj['diabetes'])
rule10 = ctrl.Rule(glucose['poor'] & insulin['good'], obj['diabetes'])
rule11 = ctrl.Rule(bloodPressure['average'] & skinThickness['average'], obj['no_diabetes'])
rule12 = ctrl.Rule(insulin['poor'] & bmi['poor'], obj['diabetes'])
rule13 = ctrl.Rule(diabetesPedigreeFunction['good'] & age['good'], obj['no_diabetes'])
rule14 = ctrl.Rule(glucose['poor'] & bloodPressure['poor'] & skinThickness['poor'], obj['diabetes'])

rules = ctrl.ControlSystem(
    [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14])

system = ctrl.ControlSystemSimulation(rules)

for i in range(len(x_train)):
    system.input['Pregnancies'] = x_train.iloc[i, 0]
    system.input['Glucose'] = x_train.iloc[i, 1]
    system.input['BloodPressure'] = x_train.iloc[i, 2]
    system.input['SkinThickness'] = x_train.iloc[i, 3]
    system.input['Insulin'] = x_train.iloc[i, 4]
    system.input['BMI'] = x_train.iloc[i, 5]
    system.input['DiabetesPedigreeFunction'] = x_train.iloc[i, 6]
    system.input['Age'] = x_train.iloc[i, 7]
    system.compute()
    pred.append(int(system.output['Outcome']))


correct_pred = 0
for i in range(len(x_test)):
    system.input['Pregnancies'] = x_test.iloc[i, 0]
    system.input['Glucose'] = x_test.iloc[i, 1]
    system.input['BloodPressure'] = x_test.iloc[i, 2]
    system.input['SkinThickness'] = x_test.iloc[i, 3]
    system.input['Insulin'] = x_test.iloc[i, 4]
    system.input['BMI'] = x_test.iloc[i, 5]
    system.input['DiabetesPedigreeFunction'] = x_test.iloc[i, 6]
    system.input['Age'] = x_test.iloc[i, 7]
    system.compute()
    pred.append(int(system.output['Outcome']))

    predicted_outcome = system.output['Outcome']
    if predicted_outcome >= 0.5:
        outcome = 1
    else:
        outcome = 0

    if outcome == y_test.iloc[i]:
        correct_pred += 1

print(correct_pred / len(y_test))
