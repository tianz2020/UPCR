# -*- coding: utf-8 -*-


class CopyScheduler:
    def __init__(self, origin_lambda, mini_lambda, n_step, s_step=0):
        #  5 1 20000 0
        self.origin_lambda = origin_lambda
        self.mini_lambda = mini_lambda
        self.n_step = n_step
        self.s_step = s_step
        self.step_interval = (self.mini_lambda - self.origin_lambda) / self.n_step

    def step_on(self, advance=True):
        if advance:
            self.s_step += 1
        if self.s_step >= self.n_step:
            return self.mini_lambda
        return self.s_step * self.step_interval + self.origin_lambda

    def dump(self):
        self_dict = {
            "origin_lambda": self.origin_lambda,
            "mini_lambda": self.mini_lambda,
            "n_step": self.n_step,
            "s_step": self.s_step,
        }
        return self_dict

    @staticmethod
    def load(self_dict):
        return CopyScheduler(self_dict["origin_lambda"],
                             self_dict["mini_lambda"],
                             self_dict["n_step"],
                             self_dict["s_step"])

    def self_load(self, self_dict):
        self.origin_lambda = self_dict["origin_lambda"]
        self.mini_lambda = self_dict["mini_lambda"]
        self.n_step = self_dict["n_step"]
        self.s_step = self_dict["s_step"]
        self.step_interval = (self.mini_lambda - self.origin_lambda) / self.n_step
