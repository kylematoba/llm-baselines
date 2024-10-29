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
    loss = loss_fun(pre_logits, y)

    to_diff = lambda _: torch.autograd.grad(loss, _, retain_graph=True)
    direct_computation = torch.autograd.grad(to_diff, model[0].weight, retain_graph=True)
    # using_hess =
    print("False")
