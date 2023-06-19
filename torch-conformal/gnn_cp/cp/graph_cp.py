from sklearn.metrics import accuracy_score
import torch
import numpy as np
import pandas as pd
from gnn_cp.data.data_manager import GraphDataManager
from gnn_cp.cp.transformations import APSTransformation, TPSTransformation, MarginTransformation


class GraphCP(object):
    def __init__(self, transformation_sequence: list, model=None, coverage_guarantee=0.9):
        self.transformation_sequence = transformation_sequence
        self.model = model
        self.coverage_guarantee = coverage_guarantee
        self.cached_scores = None
        self.score_quantile = None


    # region Basic Functions:: Scoring
    def get_scores_from_logits(self, logits, test_idx=None):
        res = torch.clone(logits)
        for transformation_item in self.transformation_sequence:
            res = transformation_item.pipe_transform(res)
        if test_idx is not None:
            res = res[test_idx]
        return res

    def get_scores(self, X, test_idx=None, model=None):
        if model is None:
            model = self.model
        logits = model.predict(X)
        result = self.get_scores_from_logits(logits)
        if test_idx is not None:
            return result[test_idx]
        return result

    # region Basic Functions:: Calibration
    def calibrate_from_logits(self, logits, y_true_mask):
        scores = self.get_scores_from_logits(logits)
        quantile_val = self.calibrate_from_scores(scores, y_true_mask)
        return quantile_val

    def calibrate_from_scores(self, scores, y_true_mask):
        score_points = scores[y_true_mask]
        quantile_idx = self.get_quantile_idx(n_points=score_points.shape[0])
        sorted_scores = torch.sort(score_points)[0]
        self.cached_scores = sorted_scores
        self.score_quantile = self.cached_scores[quantile_idx].item()
        return self.score_quantile

    def calibrate(self, X, y_true_mask, test_idx=None, model=None, y_overall=False):
        if test_idx is not None and y_overall:
            true_mask = y_true_mask[test_idx]
        else:
            true_mask = y_true_mask
        scores = self.get_scores(X, test_idx, model)
        quantile_val = self.calibrate_from_scores(scores, true_mask)
        return quantile_val
    # endregion

    # region Basic FunctionsL:: Utils
    def change_coverage_guarantee(self, new_coverage_guarantee):
        self.coverage_guarantee = new_coverage_guarantee
        if self.cached_scores is not None:
            quantile_idx = self.get_quantile_idx(n_points=self.cached_scores.shape[0])
            self.score_quantile = self.cached_scores[quantile_idx].item

    def get_quantile_idx(self, n_points):
        alpha = 1 - self.coverage_guarantee
        q_idx = int((n_points - 1) * alpha)
        return q_idx
    # endregion

    # region Basic Functions:: Prediction
    def predict_from_scores(self, scores):
        result = scores > self.score_quantile
        return result

    def predict_from_logits(self, logits):
        scores = self.get_scores_from_logits(logits)
        return self.predict_from_scores(scores)
    # endregion

    # region builtin cps
    @classmethod
    def aps_graph_cp(cls, coverage_guarantee=0.9, model=None):
        return cls(transformation_sequence=[
            APSTransformation(softmax=True)
        ], coverage_guarantee=coverage_guarantee, model=model)
    @classmethod
    def tps_graph_cp(cls, coverage_guarantee=0.9, model=None):
        return cls(transformation_sequence=[
            TPSTransformation(softmax=True)
        ], coverage_guarantee=coverage_guarantee, model=model)
    @classmethod
    def margin_graph_cp(cls, coverage_guarantee=0.9, model=None):
        return cls(transformation_sequence=[
            MarginTransformation(softmax=True)
        ], coverage_guarantee=coverage_guarantee, model=model)

    # endregion

    # region Metric Functions
    @staticmethod
    def average_set_size(prediction_set):
        set_size_vals = prediction_set.sum(axis=1)
        result = set_size_vals[set_size_vals != 0].float().mean()
        return result.item()

    @staticmethod
    def coverage(prediction_set, y_true_mask):
        cov = (prediction_set[y_true_mask].sum() / y_true_mask.sum()).item()
        return cov

    @staticmethod
    def argmax_accuracy(scores, y_true):
        y_pred = scores.int().argmax(axis=1)
        y_true_idx = y_true.int().argmax(axis=1)
        res = accuracy_score(
            y_true=y_true_idx.cpu().numpy(),
            y_pred=y_pred.cpu().numpy()
        )
        return res
    # endregion

    # region Unit Test Modules
    def shuffle_test_from_scores(self,
                     scores, y_true_mask,
                     metric=lambda pred_set, true_mask: GraphCP.coverage(pred_set, true_mask),
                     aggregation=lambda lst: (np.mean(lst), np.std(lst)),
                     n_iters=10):

        result_list = []
        for iter_idx in range(n_iters):
            calib_scores, eval_scores, calib_ymask, eval_ymask = GraphDataManager.train_test_split(
                scores, y_true_mask, training_fraction=0.5, return_idx=False)
            self.calibrate_from_scores(calib_scores, calib_ymask)
            pred_set = self.predict_from_scores(eval_scores)
            result_val = metric(pred_set, eval_ymask)
            result_list.append(result_val)
        if aggregation is None:
            return result_list
        else:
            return aggregation(result_list)

    def shuffle_test_multiple_metrics(self,
        scores, y_true_mask, aggregation=lambda lst: (np.mean(lst), np.std(lst)),
        n_iters=10, 
        metrics_dict={"set_size": lambda pred_set, true_mask: GraphCP.average_set_size(pred_set),
                      "coverage": lambda pred_set, true_mask: GraphCP.coverage(pred_set, true_mask),
                      "argmax_accuracy": lambda pred_set, true_mask: GraphCP.argmax_accuracy(pred_set, true_mask)},
        calib_fraction=0.5):
        
        result_df = []
        for iter_idx in range(n_iters):
            iteration_series = pd.Series({"attempt": iter_idx})
            calib_scores, eval_scores, calib_ymask, eval_ymask = GraphDataManager.train_test_split(
                scores, y_true_mask, training_fraction=calib_fraction, return_idx=False)
            self.calibrate_from_scores(calib_scores, calib_ymask)
            pred_set = self.predict_from_scores(eval_scores)
            for metric_name, metric_func in metrics_dict.items():
                result_val = metric_func(pred_set, eval_ymask)
                iteration_series[metric_name] = result_val
            result_df.append(iteration_series)

        result_df = pd.DataFrame(result_df)
        return result_df

    def shuffle_test_over_coverage(self,
                     scores, y_true_mask,
                     coverage_range,
                     metric=lambda pred_set, true_mask: GraphCP.coverage(pred_set, true_mask),
                     n_iters=10):
        result_df = []
        for coverage_val in coverage_range:
            self.coverage_guarantee = coverage_val
            results = self.shuffle_test_from_scores(
                scores=scores, y_true_mask=y_true_mask,
                metric=metric, aggregation=None, n_iters=n_iters
            )
            res_series = pd.DataFrame(
                [{
                    "coverage": coverage_val,
                    "attempt": i,
                    "metric": res
                } for i, res in enumerate(results)
            ])
            result_df.append(res_series)

        result_df = pd.concat(result_df).reset_index().drop(columns="index")
        return result_df


    def shuffle_metrics_over_coverage(
        self, scores, y_true_mask, coverage_range, n_iters=10,
        metrics_dict={"set_size": lambda pred_set, true_mask: GraphCP.average_set_size(pred_set),
                      "coverage": lambda pred_set, true_mask: GraphCP.coverage(pred_set, true_mask),
                      "argmax_accuracy": lambda pred_set, true_mask: GraphCP.argmax_accuracy(pred_set, true_mask)},
        calib_fraction=0.5):
        result_df = []
        for coverage_val in coverage_range:
            self.coverage_guarantee = coverage_val
            results = self.shuffle_test_multiple_metrics(
                scores=scores, y_true_mask=y_true_mask, n_iters=n_iters, metrics_dict=metrics_dict, calib_fraction=calib_fraction)
            results["coverage_guarantee"] = coverage_val
            result_df.append(results)

        result_df = pd.concat(result_df).reset_index().drop(columns="index")
        return result_df

    def shuffle_metrics_over_coverage_shifted(
        self, scores, shifted_scores, y_true_mask, coverage_range, n_iters=10,
        metrics_dict={"set_size": lambda pred_set, true_mask: GraphCP.average_set_size(pred_set),
                      "coverage": lambda pred_set, true_mask: GraphCP.coverage(pred_set, true_mask),
                      "argmax_accuracy": lambda pred_set, true_mask: GraphCP.argmax_accuracy(pred_set, true_mask)},
        calib_fraction=0.5):
        result_df = []
        for coverage_val in coverage_range:
            self.coverage_guarantee = coverage_val
            results = self.shuffle_multiple_metrics_dist_shift(
                scores=scores, shifted_scores=shifted_scores, y_true_mask=y_true_mask, n_iters=n_iters, metrics_dict=metrics_dict, calib_fraction=calib_fraction)
            results["coverage_guarantee"] = coverage_val
            result_df.append(results)

        result_df = pd.concat(result_df).reset_index().drop(columns="index")
        return result_df


    def shuffle_multiple_metrics_dist_shift(self,
        scores, shifted_scores, y_true_mask, aggregation=lambda lst: (np.mean(lst), np.std(lst)),
        n_iters=10, 
        metrics_dict={"set_size": lambda pred_set, true_mask: GraphCP.average_set_size(pred_set),
                      "coverage": lambda pred_set, true_mask: GraphCP.coverage(pred_set, true_mask),
                      "argmax_accuracy": lambda pred_set, true_mask: GraphCP.argmax_accuracy(pred_set, true_mask)},
        calib_fraction=0.5):
        
        result_df = []
        for iter_idx in range(n_iters):
            iteration_series = pd.Series({"attempt": iter_idx})
            calib_idx, eval_idx = GraphDataManager.train_test_split(
                scores, y_true_mask, training_fraction=calib_fraction, return_idx=True)

            calib_scores_clean = scores[calib_idx]
            calib_scores_shifted = shifted_scores[calib_idx]
            calib_ymask = y_true_mask[calib_idx]

            eval_scores_clean = scores[eval_idx]
            eval_scores_shifted = shifted_scores[eval_idx]
            eval_ymask = y_true_mask[eval_idx]

            self.calibrate_from_scores(calib_scores_clean, calib_ymask)
            pred_set = self.predict_from_scores(eval_scores_shifted)
            for metric_name, metric_func in metrics_dict.items():
                result_val = metric_func(pred_set, eval_ymask)
                iteration_series[metric_name] = result_val
            result_df.append(iteration_series)

        result_df = pd.DataFrame(result_df)
        return result_df