from openevolve.test_case import TestCase
import cloudpickle

cases = []
for i in range(5):
    cases.append(TestCase(args=[], kwargs={}))

cloudpickle.dump(cases, open("test_cases.pickle", "wb"))