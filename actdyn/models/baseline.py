class ARX:
    def __init__(self, order=1):
        self.order = order
        self.A = None
        self.B = None

    def fit(self, y, u):
        # y: (B, T, D), u: (B, T, U)
        B, T, D = y.shape
        U = u.shape[-1]
        Y = []
        X = []
        for t in range(self.order, T):
            y_t = y[:, t, :].reshape(B, D)
            y_prev = y[:, t - self.order : t, :].reshape(B, D * self.order)
            u_prev = u[:, t - self.order : t, :].reshape(B, U * self.order)
            Y.append(y_t)
            X.append(torch.cat((y_prev, u_prev), dim=-1))
        Y = torch.cat(Y, dim=0)  # (B*(T-order), D)
        X = torch.cat(X, dim=0)  # (B*(T-order), D*order + U*order)
        # Solve for A and B using least squares
        X_pinv = torch.linalg.pinv(X)
        AB = X_pinv @ Y  # (D*order + U*order, D)
        self.A = AB[: D * self.order, :].reshape(self.order, D, D)  # (order, D, D)
        self.B = AB[D * self.order :, :].reshape(self.order, U, D)  # (order, U, D)

    def predict(self, y_init, u, k_step):
        # y_init: (B, order, D), u: (B, order+k_step, U)
        B, T, U = u.shape
        D = y_init.shape[-1]
        y_pred = [y for y in y_init.unbind(dim=1)]  # list of (B, D)
        for k in range(k_step):
            y_prev = y_pred[-self.order :]  # order * (B, D)
            y_prev = torch.stack(y_prev, dim=1).reshape(B, -1)  # (B, D*order)
            u_t = u[:, k : k + self.order, :].reshape(B, -1)  # (B, U*order)
            y_t = y_prev @ self.A.reshape(-1, D) + u_t @ self.B.reshape(-1, D)  # (B, D)
            y_pred.append(y_t)
        y_pred = torch.stack(y_pred[self.order - 1 :], dim=1)  # (B, k_step, D)
        return y_pred
