class TokenBudget:

    DAILY_LIMIT = 200000

    def allow(self, used):
        return used < self.DAILY_LIMIT
