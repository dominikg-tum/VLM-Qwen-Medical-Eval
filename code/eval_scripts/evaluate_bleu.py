from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# # Example MRI slice descriptions
# references = [
#     "The axial T2-weighted MRI slice demonstrates normal gray-white matter differentiation in the cerebral hemispheres",
#     "Sagittal T1 image shows the brainstem and cerebellum with no apparent abnormalities",
#     "Coronal FLAIR sequence reveals bilateral ventricles of normal size and symmetry"
# ]

# candidates = [
#     "This axial T2 MRI shows normal gray and white matter in the brain hemispheres",
#     "The sagittal T1 demonstrates normal brainstem and cerebellar structures",
#     "Coronal FLAIR image shows normal symmetric ventricles"
# ]


def evaluate_bleu(references, candidates, meta=None):
    """
    Returns a list of dicts with BLEU-1/2/3/4 for each pair, suitable for saving as JSON or converting to DataFrame.
    Optionally includes meta info (e.g., case_id, image) if provided as a list of dicts.
    """
    results = []
    for i, (ref, cand) in enumerate(zip(references, candidates)):
        ref_tokens = word_tokenize(ref)
        cand_tokens = word_tokenize(cand)
        
        # Calculate BLEU scores with different n-gram weights
        bleu1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0))
        bleu2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0))
        bleu3 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        
        entry = {
            "index": i,
            "reference": ref,
            "candidate": cand,
            "bleu1": bleu1,
            "bleu2": bleu2,
            "bleu3": bleu3,
            "bleu4": bleu4
        }
        if meta is not None and i < len(meta):
            # Add meta fields such as case_id and image if present
            for k in ["case_id", "image"]:
                if k in meta[i]:
                    entry[k] = meta[i][k]
        results.append(entry)
    return results

# Example: Convert results to DataFrame and JSON
# import pandas as pd, json
# results = evaluate_bleu(references, candidates)
# df = pd.DataFrame(results)
# df.to_json("bleu_results.json", orient="records", indent=2)
# df2 = pd.read_json("bleu_results.json")
# print(df2.head())



