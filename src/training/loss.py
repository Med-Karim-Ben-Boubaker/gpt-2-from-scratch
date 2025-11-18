import torch


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss, count = 0.0, 0
    if len(data_loader) == 0:
        return float("nan")
    for i, (x, y) in enumerate(data_loader):
        if num_batches is not None and i >= num_batches:
            break
        with torch.no_grad():
            loss = calc_loss_batch(x, y, model, device)
        total_loss += loss.item()
        count += 1
    return total_loss / max(1, count)


def calc_masked_loss_batch(input_list, target_list, loss_mask_list, model, device):
    """
    Calculate loss for instruction fine-tuning with loss masking.
    
    Args:
        input_list: List of input sequences (variable lengths)
        target_list: List of target sequences (variable lengths)
        loss_mask_list: List of boolean masks indicating which tokens to include in loss
        model: The model to evaluate
        device: Device to run on
        
    Returns:
        Masked loss value
    """
    total_loss = 0.0
    total_masked_tokens = 0
    
    for input_seq, target_seq, loss_mask in zip(input_list, target_list, loss_mask_list):
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        loss_mask = loss_mask.to(device)
        
        # Use the full sequence (target_seq) for predictions, not just the input
        # The model should predict the next token for each position in the full sequence
        logits = model(target_seq.unsqueeze(0))  # Add batch dimension
        
        # Calculate cross-entropy loss for all tokens
        # We need to shift the targets by 1 position for next-token prediction
        logits = logits.squeeze(0)[:-1]  # Remove last logit (no target for it)
        targets = target_seq[1:]  # Shift targets by 1 (predict next token)
        mask = loss_mask[1:]  # Shift mask by 1 as well
        
        loss = torch.nn.functional.cross_entropy(
            logits, targets, reduction='none'
        )
        
        # Apply mask and accumulate loss
        masked_loss = loss * mask.float()
        total_loss += masked_loss.sum()
        total_masked_tokens += mask.sum()
    
    return total_loss / max(1, total_masked_tokens)


def calc_masked_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate average masked loss over a data loader.
    
    Args:
        data_loader: DataLoader yielding (input_list, target_list, loss_mask_list) tuples
        model: The model to evaluate
        device: Device to run on
        num_batches: Maximum number of batches to process
        
    Returns:
        Average masked loss
    """
    total_loss, count = 0.0, 0
    if len(data_loader) == 0:
        return float("nan")
    
    for i, (input_list, target_list, loss_mask_list) in enumerate(data_loader):
        if num_batches is not None and i >= num_batches:
            break
        with torch.no_grad():
            loss = calc_masked_loss_batch(input_list, target_list, loss_mask_list, model, device)
        total_loss += loss.item()
        count += 1
    
    return total_loss / max(1, count)
