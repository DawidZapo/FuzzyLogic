import numpy as np
import skfuzzy as fuzz
# import pandas as pd
from skfuzzy import control as ctrl

# System rozmyty dla decydowania jaka jest jakosc zycia
# bioraca pod uwage dzisiejsze naslonecznienie oraz skazenei powietrza

# rozmycie
# poprzednik - tylko w bloku rozmycia
insolation = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'insolation')
pollution = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'pollution')

insolation.automf(3)
insolation.view()

pollution['poor'] = fuzz.trimf(pollution.universe, [0, 0.3, 0.7])
pollution['good'] = fuzz.trimf(pollution.universe, [0.4, 0.7, 1])
pollution['average'] = fuzz.trimf(pollution.universe, [0.1, 0.6, 1])
pollution.view()

obj = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'quality')
obj.automf(3)
obj.view()

rule1 = ctrl.Rule(insolation['poor'] & pollution['poor'], obj['poor'])
rule2 = ctrl.Rule(insolation['average'] & pollution['poor'], obj['poor'])
rule3 = ctrl.Rule(insolation['poor'] & pollution['average'], obj['average'])
rule4 = ctrl.Rule(insolation['average'] & pollution['average'], obj['average'])
rule5 = ctrl.Rule(insolation['good'] & pollution['average'], obj['average'])
rule6 = ctrl.Rule(insolation['average'] & pollution['good'], obj['average'])
rule7 = ctrl.Rule(insolation['good'] & pollution['poor'], obj['average'])
rule8 = ctrl.Rule(insolation['good'] & pollution['good'], obj['good'])
rule9 = ctrl.Rule(insolation['good'] & pollution['average'], obj['good'])

rules = ctrl.ControlSystem([rule1, rule2, rule3])

system = ctrl.ControlSystemSimulation(rules)

system.input['insolation'] = 0.1
system.input['pollution'] = 0.1
system.compute()
print(system.output()['quality'])

obj.view()

# utworzenie zbioru rozmtytego dla cukrzykow
# siec neuronowa dla curkzykow

#np
# age = ctrl.Antecedent(np.array_equal(0, 100, 1), 'age')
# age.automf(3)
# obj.automf(2) minium dwie klasy chory i zdrowy
# tworzymy reguly
# y_pred=[]
# for i in range(len(X_test)):
#     system.input['age']= X_test.iloc[i][6] 6 to kolumna z wiekiem
#     tak dalej dla kazdej kolumny
#     system.compute()
#     y_pred.append(int(system.output['quality']))
