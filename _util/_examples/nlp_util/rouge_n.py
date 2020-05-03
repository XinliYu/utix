# examples for rouge score calculations
from _util.nlp_util import rouge_n, rouge_n_batch
from nltk.tokenize import word_tokenize

hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to help students with reading comprehension and vocabulary use the weekly news quiz to test your knowledge of story s you saw on cnn student news"

reference = "this page includes the show transcript use the transcript to help students with reading comprehension and vocabulary at the bottom of the page, " \
            "comment for a chance to be mentioned on cnn student news. " \
            "you must be a teacher or a student age # # or older to request a mention on the cnn student news roll call. " \
            "the weekly news quiz tests students' knowledge of even ts in the news"

tokenized_hypothesis = word_tokenize(hypothesis)
tokenized_reference = word_tokenize(reference)

print(rouge_n_batch([tokenized_hypothesis] * 100, [tokenized_reference] * 100))
