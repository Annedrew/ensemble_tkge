import json

class MetricCalculator():
    def __init__(self):
        self.hit1 = {}
        self.hit3 = {}
        self.hit10 = {}
        self.mr = {}
        self.mrr = {}
        self.num_facts = {}
    

    def init_emb(self, emb_name):
        if emb_name not in self.hit1.keys():
            self.hit1[emb_name] = 0
        elif emb_name not in self.hit3.keys():
            self.hit3[emb_name] = 0
        elif emb_name not in self.hit10.keys():
            self.hit10[emb_name] = 0
        elif emb_name not in self.mr.keys():
            self.mr[emb_name] = 0
        elif emb_name not in self.mrr.keys():
            self.mrr[emb_name] = 0
        elif emb_name not in self.num_facts.keys():
            self.num_facts[emb_name] = 0


    def calculate_metric(self, ranks):
        for emb_name in ranks.keys():
            self.num_facts = {emb_name: len(ranks[emb_name])}
            self.hit1[emb_name] = sum(rank for rank in ranks[emb_name] if rank == 1) / self.num_facts[emb_name]
            self.hit3[emb_name] = sum(rank for rank in ranks[emb_name] if rank == 3) / self.num_facts[emb_name]
            self.hit10[emb_name] = sum(rank for rank in ranks[emb_name] if rank == 10) / self.num_facts[emb_name]
            self.mr[emb_name] = sum(ranks[emb_name]) / self.num_facts[emb_name]
            self.mrr[emb_name] = sum((1/rank) for rank in ranks[emb_name]) / self.num_facts[emb_name]
            
            data = [self.hit1, self.hit3, self.hit10, self.mr, self.mrr]

        ret_dict = {}
        for embedding in self.hit1.keys():
            ret_dict[embedding] = {
                "HIT1": self.hit1[embedding], 
                "HIT3": self.hit3[embedding], 
                "HIT10": self.hit10[embedding], 
                "MR": self.mr[embedding], 
                "MRR": self.mrr[embedding]
            }
        
        # with open("overall.json", "w") as f:
        #     json.dump(ret_dict, f, indent=4)

        return ret_dict