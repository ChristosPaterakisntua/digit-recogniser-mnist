import torch


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def ask_yes_or_no(prompt: str) -> bool:
    answers = ["y", "n"]
    prompt = prompt.strip() + " (y/n): "
    ans = ""
    while ans not in answers:
        ans = input(prompt).lower().strip()
    return ans == "y"


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    All values are normalized to 0.0 - 1.0 using z score method

    Args
    ------
    tensor : torch.Tensor
        the input tensor

    Returns
    ------
    torch.Tensor
        normalized tensor

    Note
    ------
    `y = (x - mean) / std` where:
    - x: input tensor
    - y: output tensor
    """
    tensor = tensor.to(dtype=torch.float32)
    mean, std = torch.mean(tensor), torch.std(tensor)
    tensor = (tensor - mean) / std
    return tensor.to(dtype=torch.float32)
