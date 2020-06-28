
from imblearn.over_sampling import SMOTE
class Sampler:
    def __init__(self, sampler, random_state, data):
        self.data=data
        self.random_state=random_state
        self.sampler=sampler
    def run(self):
        if self.sampler == "smote":
            sm = SMOTE(random_state=self.random_state)
            size=len(self.data["x_train"])
            pixel=self.data["x_train"].shape[1]
            X_res, y_res = sm.fit_resample(self.data["x_train"].reshape((size, pixel**2)), self.data["y_train"])
            ans = self.data.copy()
            ans["x_train"] = X_res.reshape((len(X_res), pixel, pixel, 1))
            ans["y_train"] = y_res
            return ans
