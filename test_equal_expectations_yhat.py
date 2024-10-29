import torch

if __name__ == "__main__":
    b = 19
    d0 = 4
    d1 = 7
    d2 = 9

    y = torch.nn.functional.one_hot(torch.randint(0, d2, size=(b,)), num_classes=d2).to(float)
    x = torch.randn(b, d0)

    model = torch.nn.Sequential(torch.nn.Linear(d0, d1),
                                torch.nn.GELU(),
                                torch.nn.Linear(d1, d2))
    loss_fun = torch.nn.CrossEntropyLoss()
    pre_logits = model(x)
    logits = torch.nn.Softmax(1)(pre_logits)
    # logits2 = torch.softmax(pre_logits, dim=0)

    loss = loss_fun(pre_logits, y)
    params = [model.parameters()]
    outs = torch.autograd.grad(loss, inputs=model.parameters(), retain_graph=True)
    loss.backward(retain_graph=True)

    num_trials = 200
    vals = [None] * num_trials
    for idx in range(num_trials):
        yhat = torch.nn.functional.one_hot(torch.multinomial(logits, num_samples=1).flatten(),
                                           num_classes=d2).to(float)
        idx_loss = loss_fun(pre_logits, yhat)
        idx_outs = torch.autograd.grad(idx_loss, inputs=model.parameters(), retain_graph=True)
        vals[idx] = idx_outs[0].T @ idx_outs[0]
    avged_vals = sum(vals) / num_trials

    loss_gold = loss_fun(pre_logits, logits)
    outs_gold = torch.autograd.grad(loss_gold, inputs=model.parameters(), retain_graph=True)
    gtg_gold = outs_gold[0].T @ outs_gold[0]
    # gold_loss_val = loss_fun(pre_logits, logits)
    # gold_loss_val.backward(retain_graph=True)
    # g = model[0].weight.grad
    # gtg = g.T @ g
    #
    # num_trials = 5000
    # vals = [None] * num_trials
    # for idx in range(num_trials):
    #     yhat = torch.nn.functional.one_hot(torch.multinomial(logits, num_samples=1).flatten(),
    #                                        num_classes=d2).to(float)
    #     losshat = loss_fun(pre_logits, yhat)
    #     losshat.backward(retain_graph=True)
    #     g = model[0].weight.grad
    #     vals[idx] = g.T @ g
    # avged_vals = sum(vals) / num_trials
    # #
    # # loss.backward(retain_graph=True)
    # # g = model[0].weight.grad
    # # print(g.T @ g)
    # # print(avged_vals)
    #
    print("Done")
