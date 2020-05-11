import pandas as pd
import numpy as np
import cvxpy as cv

class Optimiser:

    def __init__(self, initial_portfolio, turnover, max_wt, longshort=True):
        self.symbols = np.array(initial_portfolio.index)
        self.init_wt = np.array(initial_portfolio['weight'])
        self.opt_wt = cv.Variable(self.init_wt.shape)
        self.alpha = np.array(initial_portfolio['alpha'])
        self.longshort = longshort
        self.turnover = turnover
        self.max_wt = max_wt
        if self.longshort:
            self.min_wt = -self.max_wt
            self.net_exposure = 0
            self.gross_exposure = 1
        else:
            self.min_wt = 0
            self.net_exposure = 1
            self.gross_exposure = 1

    def optimise(self):
        constraints = self.get_constraints()
        optimisation = cv.Problem(cv.Maximize(cv.sum(self.opt_wt*self.alpha)), constraints)
        optimisation.solve()
        status = optimisation.status
        if status == 'optimal':
            optimal_portfolio = pd.Series(np.round(
                optimisation.solution.primal_vars[list(
                    optimisation.solution.primal_vars.keys())[0]], 3), index=self.symbols)
        else:
            optimal_portfolio = pd.Series(np.round(self.init_wt, 3), index=self.symbols)
        return optimal_portfolio, status

    def get_constraints(self):
        min_wt = self.opt_wt >= self.min_wt
        max_wt = self.opt_wt <= self.max_wt
        turnover = cv.sum(cv.abs(self.opt_wt-self.init_wt)) <= self.turnover*2
        net_exposure = cv.sum(self.opt_wt) == self.net_exposure
        gross_exposure = cv.sum(cv.abs(self.opt_wt)) <= self.gross_exposure
        return [min_wt, max_wt, turnover, net_exposure,gross_exposure]

def run_optimisation():
    initial_portfolio = pd.DataFrame({
        'symbol': ['AAPL','MSFT', 'GGOGL', 'TSLA'],
        'weight': [-0.25, -0.25, 0.25, 0.25],
        'alpha': [-0.2, -0.3, 0.25, 0],
    }).set_index('symbol')

    opt = Optimiser(initial_portfolio, turnover=0.2, max_wt=0.3, longshort=True)
    result, status = opt.optimise()
    print()

run_optimisation()