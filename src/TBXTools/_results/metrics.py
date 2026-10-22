class Metrics:

    def __init__(self):
        self.eval_data = None
        self.processor = None

    def score(self, pred_dict, true_dict):
        tp = 0
        total_preds = 0
        total_trues = 0

        for i in true_dict.keys():
            preds = pred_dict.get(i, [])
            trues = true_dict.get(i, [])

            total_preds += len(preds)
            total_trues += len(trues)

            true_pool = list(trues)

            for pred in preds:
                if pred in true_pool:
                    tp += 1
                    true_pool.remove(pred)

        precision = tp / total_preds if total_preds > 0 else 0.0
        recall = tp / total_trues if total_trues > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1

    def compute_metrics(self, p):
        import numpy as np

        prediction_logits, label_ids = p
        predictions = np.argmax(prediction_logits, axis=2)

        pred_dict = {}
        true_dict = {}

        for segment_idx in range(len(self.eval_data)):
            tokens = self.eval_data[segment_idx]["tokens"]
            predicted_ids = predictions[segment_idx]
            true_ids = label_ids[segment_idx]

            pred_terms = self.processor._bio_to_terms(tokens, predicted_ids)
            true_terms = self.processor._bio_to_terms(tokens, true_ids)

            pred_dict[segment_idx] = pred_terms
            true_dict[segment_idx] = true_terms

        precision, recall, f1 = self.score(pred_dict, true_dict)

        return {"precision": precision, "recall": recall, "f1": f1}

    def compute_metrics_with_golden_labels(self, candidate_terms, gl):
        pred_list = candidate_terms
        true_list = list(gl.values())

        pred_dict = {i: terms for i, terms in enumerate(pred_list)}
        true_dict = {i: terms for i, terms in enumerate(true_list)}

        precision, recall, f1 = self.score(pred_dict, true_dict)

        print(
            f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}")
        return {"precision": precision, "recall": recall, "f1": f1}


    # work in progress
    def analyze_false_positives(self, candidate_terms, gl, save=False):
        from collections import Counter

        pred_list = candidate_terms
        true_list = list(gl.values())

        false_positives = []

        for preds, trues in zip(pred_list, true_list):

            true_pool = [str(true).lower().strip() for true in trues]

            for pred in preds:
                pred_norm = str(pred).lower().strip()

                if pred_norm not in true_pool:
                    false_positives.append(pred)

                else:
                    true_pool.remove(pred_norm)

        fp_counts = Counter(false_positives)

        print(f"\nTop 30 false positives")
        for term, count in fp_counts.most_common(30):
            print(f"{term} \t {count}")

        # if save:
        #     for term, count in fp_counts.most_common():
        #         print(f"{term}\t{count}")

    def _old_compute_metrics_lemm(self, p):
        import numpy as np
        import pandas as pd
        # label_ids are true padded label IDs from eval_data
        prediction_logits, label_ids = p
        predictions = np.argmax(prediction_logits, axis=2)
        all_pred_terms = []
        all_true_terms = []

        rows = []
        for i in range(len(self.eval_data)):
            tokens = self.eval_data[i]["tokens"]
            predicted_ids = predictions[i]
            true_ids = label_ids[i]
            # for j, (token, true_id, pred_id) in enumerate(zip(tokens, true_ids, predicted_ids)):
            #     if token in ["[CLS]", "[SEP]", "[PAD]"] or true_id == -100:
            #         continue

            #     rows.append({
            #         "segment_id": i,
            #         "token_index": j,
            #         "token": token,
            #         "true_label": self.processor._id2label[true_id],
            #         "pred_label": self.processor._id2label[pred_id],
            #         "correct": true_id == pred_id
            #     })

            reconstructed_predicted_terms = self.processor._bio_to_terms(
                tokens, predicted_ids)
            all_pred_terms.extend(reconstructed_predicted_terms)
            reconstructed_true_terms = self.processor._bio_to_terms(
                tokens, true_ids)
            all_true_terms.extend(reconstructed_true_terms)

            # print(f"Segment {i}")
            # print("pred terms in segment:", reconstructed_predicted_terms)
            # print("true terms in segment:", reconstructed_true_terms)

        # df = pd.DataFrame(rows)
        # df.to_csv("debug_test.csv", index=False)

        precision, recall, f1 = self.score(all_pred_terms, all_true_terms)
        # print coincidences between preds and true terms to check what is being annotated as true terms, or check true terms, true terms are problematic since they are also automatically annotated not some golden labels... need to check asap

        # check label alignment, what is being appended to sqlite is bert tokens im pretty sure, check engitech

        # common = list(set(all_pred_terms) & set(all_true_terms))
        # print(common)
        # print(all_pred_terms)

        return {"precision": precision, "recall": recall, "f1": f1}
