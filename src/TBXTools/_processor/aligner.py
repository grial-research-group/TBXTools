import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

class Aligner:

    def align_words(self, src_list, tgt_list, model="sentence-transformers/LaBSE",threshold=0.75, synonym=False, score=True):
        '''
        Aligns source and target word lists using sentence embeddings.

        - Round 1 (Hungarian Algorithm) guarantees an optimal 1-to-1 mapping.
        - Rounds 2 & 3 (optional, when synonym=True) recover secondary translations/synonyms.
        '''
        
        model = SentenceTransformer(model)

        # Generate contextual embeddings for both language lists
        src_vectors = model.encode(src_list, convert_to_numpy=True, show_progress_bar=False)
        tgt_vectors = model.encode(tgt_list, convert_to_numpy=True, show_progress_bar=False)

        # Vectors normalization
        src_vectors = src_vectors / np.linalg.norm(
            src_vectors, axis=1, keepdims=True
        )
        tgt_vectors = tgt_vectors / np.linalg.norm(
            tgt_vectors, axis=1, keepdims=True
        )

        # Compute the similarity matrix
        matrix_sim = np.dot(src_vectors, tgt_vectors.T)

        # ROUND 1: Strict (1-to-1)
        src_ind, tgt_ind = linear_sum_assignment(-matrix_sim)

        results_list = []
        chosen_src_indices = set()
        chosen_tgt_indices = set()

        for r, c in zip(src_ind, tgt_ind):
            sim_score = float(matrix_sim[r, c])
            if sim_score >= threshold:
                item_src = src_list[r]
                item_tgt = tgt_list[c]

                chosen_src_indices.add(r)
                chosen_tgt_indices.add(c)

                score_val = round(sim_score, 4)
                if score:
                    results_list.append((item_src, item_tgt, score_val))
                else:
                    results_list.append((item_src, item_tgt))
        
        # ROUND 2: Rematch for Synonyms / Leftovers
        # index excluded from first round
        if synonym:
            excluded_src_idx = [
                i for i in range(len(src_list)) if i not in chosen_src_indices
            ]
            excluded_tgt_idx = [
                i for i in range(len(tgt_list)) if i not in chosen_tgt_indices
            ]

            for r in excluded_src_idx:
                c_best = np.argmax(matrix_sim[r, :])
                sim_score = float(matrix_sim[r, c_best])
                if sim_score >= threshold:
                    item_src, item_tgt = src_list[r], tgt_list[c_best]
                    score_val = round(sim_score, 4)
                    entry = (
                        (item_src, item_tgt, score_val)
                        if score
                        else (item_src, item_tgt)
                    )
                    if entry not in results_list:
                        results_list.append(entry)

            for c in excluded_tgt_idx:
                r_best = np.argmax(matrix_sim[:, c])
                sim_score = float(matrix_sim[r_best, c])
                if sim_score >= threshold:
                    item_src, item_tgt = src_list[r_best], tgt_list[c]
                    score_val = round(sim_score, 4)
                    entry = (
                        (item_src, item_tgt, score_val)
                        if score
                        else (item_src, item_tgt)
                    )
                    if entry not in results_list:
                        results_list.append(entry)

            
            grouped_dict = defaultdict(list)
            for entry in results_list:
                src_item, tgt_item = entry[0], entry[1]
                score_item = entry[2] if score else None
                grouped_dict[src_item].append((tgt_item, score_item))

            flattened_results = []
            for src_item, tgt_entries in grouped_dict.items():
                seen = set()
                for tgt_item, sc in tgt_entries:
                    if tgt_item not in seen:
                        seen.add(tgt_item)
                        if score:
                            flattened_results.append((src_item, tgt_item, sc))
                        else:
                            flattened_results.append((src_item, tgt_item))
            return flattened_results

        return results_list