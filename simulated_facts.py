
from de_simple.rank_calculator import RankCalculator as DE_Rank
from TERO.rank_calculator import RankCalculator as TERO_Rank

class SimulatedRank:
    # def __init__(self, params, quads, model, embedding_name):
    #     self.params = params
    def __init__(self, quads, model, embedding_name):
        self.quads = quads
        self.model = model
        self.model.eval()
        self.embedding_name = embedding_name

    def sim_rank(self):
        ranked_quads = []
        
        if self.embedding_name in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
            # rank_calculator = DE_Rank(self.params, self.model)
            rank_calculator = DE_Rank(self.model)
        if self.embedding_name in ["TERO", "ATISE"]:
            # rank_calculator = TERO_Rank(self.params, self.model)
            rank_calculator = TERO_Rank(self.model)
            

        for i, quad in zip(range(0, len(self.quads)), self.quads):
            if i % 100 == 0:
                print("Ranking fact " + str(i) + "-" + str(i + 99) + " (total number: " + str(len(self.quads)) + ") with embedding " + self.embedding_name)

            ranked_quad = quad
            if "RANK" not in ranked_quad.keys():
                ranked_quad["RANK"] = {}

            ranked_quad["RANK"][self.embedding_name] = str(rank_calculator.get_rank_of(quad["HEAD"], quad["RELATION"],
                                                                                       quad["TAIL"], quad["TIME"],
                                                                                       quad["ANSWER"]))
            ranked_quads.append(ranked_quad)

        return ranked_quads

class SimulatedScore:
    def __init__(self, quads, model, embedding_name):
        self.quads = quads
        self.model = model
        self.model.eval()
        self.embedding_name = embedding_name

    def sim_score(self):
        ranked_quads = []
        
        if self.embedding_name in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
            rank_calculator = DE_Rank(self.model)
        if self.embedding_name in ["TERO", "ATISE"]:
            rank_calculator = TERO_Rank(self.model)
            

        for i, quad in zip(range(0, len(self.quads)), self.quads):
            if i % 100 == 0:
                print("Ranking fact " + str(i) + "-" + str(i + 99) + " (total number: " + str(len(self.quads)) + ") with embedding " + self.embedding_name)

            ranked_quad = quad
            if "RANK" not in ranked_quad.keys():
                ranked_quad["RANK"] = {}

            ranked_quad["RANK"][self.embedding_name] = str(rank_calculator.get_sim_score(quad["HEAD"], quad["RELATION"],
                                                                                       quad["TAIL"], quad["TIME"],
                                                                                       quad["ANSWER"]))
            ranked_quads.append(ranked_quad)

        return ranked_quads