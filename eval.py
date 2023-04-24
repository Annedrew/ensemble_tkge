from metric_calculator import MetricCalculator

class Eval():
    def eval(self, best_weights, rank_score):
        metric = MetricCalculator()
        ensemble_scores = []
        for i in range(rank_score.shape[1]):
            ensemble_score = 0
            for j in range(rank_score.shape[0]):
                ensemble_score += best_weights[j] * rank_score[j][i]
            ensemble_scores.append(ensemble_score)
        ens_rank = {"ens": ensemble_scores}
        # print(ens_rank)
        eval = metric.calculate_metric(ens_rank)
        eval = eval.values()
            
        return eval
