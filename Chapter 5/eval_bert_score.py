from bert_score import score

candidate = ["The cat is on the mat."]
reference = ["A cat is sitting on a mat."]

P, R, F1 = score(cands=candidate, 
                 refs=reference, 
                 model_type="roberta-large", 
                 lang="en")

print(f"Precision: {P.item()}\nRecall: {R.item()}\nF1: {F1.item()}")

from bert_score import plot_example

plot_example(candidate=candidate[0], 
             reference=reference[0], 
             model_type="roberta-large", 
             lang="en")
