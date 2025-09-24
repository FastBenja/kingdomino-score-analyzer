import sys
import os
import csv
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add directory above to path, to allow import of main
from main import ImageScore # Import main class to extend functionality


class PassFailTest(ImageScore): # Test class
    def __init__(self):
        super().__init__()
        self.total_tests = 0
        self.correct_tests = 0
        self.answers = {}
        with open("test/results.csv", newline="\n") as csvfile: # Load answers from csv-file
            reader = csv.reader(csvfile)
            for answer in reader:
                self.answers[answer[0]] = int(answer[1]) # Store answers in dictonary with filename as key and score as value

    def pass_fail_test(self): # Run pass/fail test
        print(f"\n{'='*40}\nStarting pass/fail test with {len(self.image_dict)} images\n{'='*40}\n")
        for file, img in self.image_dict.items(): # Iterate over all images
            calculated_answer = self.eval_raw_img(img) # Get calculated score
            try:
                correct_answer = self.answers[file] # Get correct score from dictonary
            except KeyError:
                print(f"⚠️ WARNING: No correct answer found for file {file}, skipping test")
                continue
            if calculated_answer == correct_answer:
                print(f"✅ PASS File: {file} calculated score {calculated_answer} expected score {correct_answer}")
                self.correct_tests += 1
            else:
                print(f"❌ FAIL File: {file} calculated score {calculated_answer} expected score {correct_answer}")
            self.total_tests += 1
        
        correct_pct = np.divide(self.correct_tests, self.total_tests) * 100
        print(f"\n{"="*40}\nPassed {self.correct_tests} out of {self.total_tests} tests ({correct_pct:.2f}%)\n{"="*40}\n")
        
            

if __name__ == "__main__":
    test = PassFailTest()
    test.pass_fail_test()
