
class Result:

    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def set_tp(self, tp):
        self.tp = tp

    def set_tn(self, tn):
        self.tn = tn

    def set_fp(self, fp):
        self.fp = fp

    def set_fn(self, fn):
        self.fn = fn

    def get_precision(self) -> float:
        return self.tp / (self.tp + self.fp)

    def get_accuracy(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

    def get_sepecificity(self) -> float:
        return self.tn / (self.tn + self.fp)

    def get_recall(self) -> float:
        return self.tp / (self.tp + self.fn)

    def get_f1_score(self) -> float:
        return 2 * ((self.get_precision() * self.get_recall()) / (self.get_precision() + self.get_recall()))


