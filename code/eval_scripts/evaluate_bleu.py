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


def evaluate_bleu(references, candidates):
    # Calculate BLEU-1,2,3,4 scores for each pair
    for i, (ref, cand) in enumerate(zip(references, candidates)):
        ref_tokens = word_tokenize(ref)
        cand_tokens = word_tokenize(cand)
        
        # Calculate BLEU scores with different n-gram weights
        bleu1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0))
        bleu2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0))
        bleu3 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        
        print(f"\nExample {i+1}:")
        print(f"Reference: {ref}")
        print(f"Candidate: {cand}")
        print(f"BLEU-1: {bleu1:.4f}")
        print(f"BLEU-2: {bleu2:.4f}")
        print(f"BLEU-3: {bleu3:.4f}")
        print(f"BLEU-4: {bleu4:.4f}")
