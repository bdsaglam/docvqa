## Submission Rules

A single JSON file is expected.
Main JSON element is an array of dictionaries with keys question_id, answer, and full_answer.
question_id: The question id as it appears in the dataset.

answer: The final answer of the model. This is the exact answer that will be evaluated.

full_answer: (optional) Full reasoning answer of the model, can be an empty string.

The submitted JSON file is automatically checked at the time of submission, and a submission log is presented to the user along with a confirmation of the submission. The checks performed are the following:

That the file submitted is a valid JSON file.
That the root element is an array.
That the number of elements are equal to the ground truth.
That the key question_id matches a question_id of the Ground Truth.

See ./submission_sample_val.json for an example of a valid submission file.

Results (Required)
A single JSON file is expected.
Main JSON element is an array of dictionaries with keys question_id, answer, and full_answer.
question_id: The question id as it appears in the dataset.

answer: The final answer of the model. This is the exact answer that will be evaluated.

full_answer: (optional) Full reasoning answer of the model, can be an empty string.

The submitted JSON file is automatically checked at the time of submission, and a submission log is presented to the user along with a confirmation of the submission. The checks performed are the following:

That the file submitted is a valid JSON file.
That the root element is an array.
That the number of elements are equal to the ground truth.
That the key question_id matches a question_id of the Ground Truth.
